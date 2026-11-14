#include <Wire.h>
#include "heartRate.h"
#include <math.h>
#include <stdint.h>
#include "MAX30105.h"
#include <WiFi.h>
#include <HTTPClient.h>

// Set to 1 to view the live PPG pulse waveform in Arduino Serial Plotter.
// Set to 0 for the one-line physiological dashboard in Serial Monitor.
#define OUTPUT_PPG_PLOT 0

// ---------------- PIN CONFIGURATION ----------------
const int SDA_PIN = 21;
const int SCL_PIN = 22;
const int GSR_PIN = 34;
const uint8_t MPU_ADDR = 0x68;

MAX30105 particleSensor;
bool maxReady = false;
bool mpuReady = false;

// ---------------- TIMING ----------------
const uint32_t MPU_PERIOD_US = 10000;       // 100 Hz
const uint32_t REPORT_PERIOD_MS = 1000;     // Dashboard every second
const uint32_t RESP_REPORT_PERIOD_MS = 5000;// Respiration refresh every 5 s
const uint32_t EDA_CALIBRATION_MS = 10000;  // Initial resting baseline period

uint32_t lastMpuSampleUs = 0;
uint32_t lastReportMs = 0;
uint32_t lastRespReportMs = 0;
uint32_t bootTimeMs = 0;

// ---------------- USER-CALIBRATABLE THRESHOLDS ----------------
// Your earlier GSR tests showed approximately 700-1400 counts with finger
// contact and >2000 counts without contact. Re-check these values on the
// final electrode/watch arrangement and adjust the two constants below.
const int GSR_CONTACT_RAW_MAX = 1800;
const bool GSR_LOWER_RAW_MEANS_HIGHER_EDA = true;
const float EDA_ENTER_HIGH_PERCENT = 12.0f;
const float EDA_EXIT_HIGH_PERCENT = 7.0f;

// Keep the GSR electrodes on the skin during the first 10 seconds.
const uint32_t IR_FINGER_THRESHOLD = 5000;
const uint32_t IR_SATURATION_LIMIT = 250000;

// Finger detection stabilisation
const uint8_t FINGER_CONFIRM_SAMPLES = 5;
const uint16_t FINGER_LOSS_SAMPLES = 75;

// PPG peak detection tuning. These may need adjustment for wrist placement.
const float PPG_DC_ALPHA = 0.010f;
const float PPG_SMOOTH_ALPHA = 0.25f;
const float PPG_MIN_PEAK_AMPLITUDE = 8.0f;
const float PPG_ENVELOPE_DECAY = 0.98f;
const uint32_t PPG_REFRACTORY_MS = 300;

// HR/HRV limits. RR intervals are measured in milliseconds.
const uint16_t MIN_RR_MS = 350;
const uint16_t MAX_RR_MS = 2000;
const int HR_HISTORY_SIZE = 4;
const int RR_HISTORY_SIZE = 30;
const int MIN_RR_FOR_HRV = 10;

// Respiration extraction from PPG baseline modulation.
// Filters assume a 100 Hz PPG sample rate, then decimate to 20 Hz.
const float RESP_LP_ALPHA = 0.0309f; // ~0.5 Hz low-pass
const float RESP_BASE_ALPHA = 0.0050f; // ~0.08 Hz baseline tracker
const int RESP_DECIMATION = 5;
const int RESP_BUFFER_SIZE = 600; // 30 seconds at 20 Hz
const int RESP_MIN_SAMPLES = 400; // At least 20 seconds of data

// MPU6500 activity thresholds. Tune against your own wrist recordings.
const float MOVE_ACCEL_RMS_G = 0.05f;
const float MOVE_GYRO_RMS_DPS = 8.0f;
const float WORKOUT_ACCEL_RMS_G = 0.18f;
const float WORKOUT_GYRO_RMS_DPS = 15.0f;

// Replace 75 with the person's measured resting HR baseline if available.
const float PERSONAL_RESTING_HR_BPM = 75.0f;
const float REST_HR_RISE_BPM = 20.0f;
const int REST_HR_SUSTAIN_SECONDS = 15;

// ---------------- GSR / EDA ACCUMULATORS ----------------
double gsrSum = 0.0;
uint32_t gsrSamples = 0;
double gsrCalibrationSum = 0.0;
uint32_t gsrCalibrationSamples = 0;
float gsrBaseline = NAN;
bool edaCalibrated = false;
bool edaHigh = false;

// ---------------- MPU ACCUMULATORS ----------------
double accXSum = 0.0, accYSum = 0.0, accZSum = 0.0;
double dynamicAccelSqSum = 0.0;
double gyroVectorSqSum = 0.0;
uint32_t mpuSamples = 0;
uint32_t mpuErrors = 0;

float lastAccelDynamicRms = NAN;
float lastGyroRms = NAN;
float lastAccX = NAN, lastAccY = NAN, lastAccZ = NAN;

// ---------------- PPG / HEART RATE ----------------
bool fingerPresent = false;
bool ppgDcReady = false;

uint8_t fingerGoodSamples = 0;
uint16_t fingerBadSamples = 0;
float ppgDc = 0.0f;
float ppgAcFiltered = 0.0f;
float ppgPrevious1 = 0.0f;
float ppgPrevious2 = 0.0f;
float ppgEnvelope = 0.0f;
uint32_t ppgSampleTimeMs = 0;
bool ppgClockReady = false;
uint32_t lastBeatMs = 0;
bool haveLastBeat = false;

float hrHistory[HR_HISTORY_SIZE];
int hrWrite = 0;
int hrCount = 0;
uint16_t rrHistory[RR_HISTORY_SIZE];
int rrWrite = 0;
int rrCount = 0;

// ---------------- PPG-DERIVED RESPIRATION ----------------
float respLowPass = 0.0f;
float respBaseline = 0.0f;
bool respFilterReady = false;
int respDecimator = 0;
float respBuffer[RESP_BUFFER_SIZE];
int respWrite = 0;
int respCount = 0;
float latestRespRate = NAN;

// ---------------- ACTIVITY STATE ----------------
enum ActivityState {
  ACT_UNKNOWN,
  ACT_STATIONARY,
  ACT_MOVING,
  ACT_WORKOUT_LIKE
};

ActivityState activityState = ACT_UNKNOWN;
int highHrAtRestSeconds = 0;
bool highHrAtRest = false;

// ---------------- MPU6500 REGISTER ACCESS ----------------
bool writeMPU(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readRegister(uint8_t address, uint8_t reg, uint8_t &value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(address, (size_t)1, true) != 1) return false;
  value = Wire.read();
  return true;
}

bool initMPU() {
  if (!writeMPU(0x6B, 0x00)) return false; // Wake up
  delay(100);
  if (!writeMPU(0x1A, 0x03)) return false; // DLPF
  if (!writeMPU(0x19, 0x09)) return false; // 1 kHz / (1 + 9) = 100 Hz
  if (!writeMPU(0x1C, 0x00)) return false; // +/-2 g
  if (!writeMPU(0x1B, 0x00)) return false; // +/-250 deg/s
  if (!writeMPU(0x1D, 0x03)) return false; // Accelerometer DLPF

  uint8_t id = 0;
  return readRegister(MPU_ADDR, 0x75, id) && id == 0x70;
}

bool readMPU(int16_t &xA, int16_t &yA, int16_t &zA,
             int16_t &xG, int16_t &yG, int16_t &zG) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (size_t)14, true) != 14) return false;

  uint8_t b[14];
  for (int i = 0; i < 14; i++) {
    if (!Wire.available()) return false;
    b[i] = Wire.read();
  }

  xA = (int16_t)((uint16_t)b[0] << 8 | b[1]);
  yA = (int16_t)((uint16_t)b[2] << 8 | b[3]);
  zA = (int16_t)((uint16_t)b[4] << 8 | b[5]);
  // b[6], b[7] are temperature bytes.
  xG = (int16_t)((uint16_t)b[8]  << 8 | b[9]);
  yG = (int16_t)((uint16_t)b[10] << 8 | b[11]);
  zG = (int16_t)((uint16_t)b[12] << 8 | b[13]);
  return true;
}

// ---------------- RESET / METRIC HELPERS ----------------
void resetMpuWindow() {
  accXSum = accYSum = accZSum = 0.0;
  dynamicAccelSqSum = 0.0;
  gyroVectorSqSum = 0.0;
  mpuSamples = 0;
  mpuErrors = 0;
}

void resetPpgAndRespiration() {
  ppgDcReady = false;
  ppgClockReady = false;
  ppgSampleTimeMs = 0;
  ppgDc = 0.0f;
  ppgAcFiltered = 0.0f;
  ppgPrevious1 = 0.0f;
  ppgPrevious2 = 0.0f;
  ppgEnvelope = 0.0f;
  haveLastBeat = false;
  lastBeatMs = 0;
  hrWrite = 0;
  hrCount = 0;
  rrWrite = 0;
  rrCount = 0;
  respFilterReady = false;
  respLowPass = 0.0f;
  respBaseline = 0.0f;
  respDecimator = 0;
  respWrite = 0;
  respCount = 0;
  latestRespRate = NAN;
  fingerGoodSamples = 0;
  fingerBadSamples = 0;
}
float medianRecentRR() {

  if (rrCount == 0) return NAN;

  int n = rrCount;
  if (n > 5) n = 5;

  uint16_t values[5];

  int start =
    (rrWrite - n + RR_HISTORY_SIZE) %
    RR_HISTORY_SIZE;

  for (int i = 0; i < n; i++) {
    values[i] =
      rrHistory[(start + i) % RR_HISTORY_SIZE];
  }

  // Sort intervals
  for (int i = 1; i < n; i++) {

    uint16_t key = values[i];
    int j = i - 1;

    while (j >= 0 && values[j] > key) {
      values[j + 1] = values[j];
      j--;
    }

    values[j + 1] = key;
  }

  if (n % 2 == 0) {
    return (values[n / 2 - 1] +
            values[n / 2]) / 2.0f;
  }

  return values[n / 2];
}
float currentHeartRate() {

  if (!fingerPresent ||
      !haveLastBeat ||
      rrCount < 3) {
    return NAN;
  }

  float medianRR = medianRecentRR();

  if (!isfinite(medianRR) || medianRR <= 0) {
    return NAN;
  }

  return 60000.0f / medianRR;
}
float currentHRV_RMSSD() {

  if (!fingerPresent ||
      !haveLastBeat ||
      rrCount < MIN_RR_FOR_HRV) {

    return NAN;
  }

  double diffSqSum = 0.0;
  int differences = 0;

  int start =
    (rrWrite - rrCount + RR_HISTORY_SIZE) %
    RR_HISTORY_SIZE;

  for (int i = 1; i < rrCount; i++) {

    int prevIndex =
      (start + i - 1) % RR_HISTORY_SIZE;

    int currIndex =
      (start + i) % RR_HISTORY_SIZE;

    int32_t diff =
      (int32_t)rrHistory[currIndex] -
      (int32_t)rrHistory[prevIndex];

    diffSqSum += (double)diff * diff;
    differences++;
  }

  if (differences == 0) {
    return NAN;
  }

  return sqrt(diffSqSum / differences);
}

bool recordBeat(uint32_t beatTimeMs) {

  if (!haveLastBeat) {

    lastBeatMs = beatTimeMs;
    haveLastBeat = true;

    return false;
  }

  uint32_t rr = beatTimeMs - lastBeatMs;

  // Reject duplicate or excessively close peaks
  if (rr < MIN_RR_MS) {
    return false;
  }

  // Resynchronise after a missed beat
  if (rr > MAX_RR_MS) {

    lastBeatMs = beatTimeMs;

    return false;
  }

  // Reject suspiciously short intervals
  // compared with the recent accepted intervals.
  if (rrCount >= 3) {

    float previousMedian = medianRecentRR();

    if (
      isfinite(previousMedian) &&
      rr < previousMedian * 0.60f
    ) {
      return false;
    }
  }

  lastBeatMs = beatTimeMs;

  rrHistory[rrWrite] = (uint16_t)rr;

  rrWrite =
    (rrWrite + 1) % RR_HISTORY_SIZE;

  if (rrCount < RR_HISTORY_SIZE) {
    rrCount++;
  }

  float bpm = 60000.0f / (float)rr;

  hrHistory[hrWrite] = bpm;

  hrWrite =
    (hrWrite + 1) % HR_HISTORY_SIZE;

  if (hrCount < HR_HISTORY_SIZE) {
    hrCount++;
  }

  return true;
}
void storeRespSample(float value) {
  respBuffer[respWrite] = value;
  respWrite = (respWrite + 1) % RESP_BUFFER_SIZE;
  if (respCount < RESP_BUFFER_SIZE) respCount++;
}

float estimateRespirationRate() {
  if (respCount < RESP_MIN_SAMPLES) return NAN;

  int n = respCount;
  int start = (respWrite - n + RESP_BUFFER_SIZE) % RESP_BUFFER_SIZE;

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += respBuffer[(start + i) % RESP_BUFFER_SIZE];
  }
  float mean = (float)(sum / n);

  double varianceSum = 0.0;
  for (int i = 0; i < n; i++) {
    float d = respBuffer[(start + i) % RESP_BUFFER_SIZE] - mean;
    varianceSum += (double)d * d;
  }
  float sd = sqrt((float)(varianceSum / n));
  if (sd < 0.5f) return NAN; // Too little respiratory modulation / poor signal.

  float deadband = sd * 0.20f;
  bool armedBelow = false;
  int risingCrossings = 0;

  for (int i = 0; i < n; i++) {
    float v = respBuffer[(start + i) % RESP_BUFFER_SIZE] - mean;
    if (v < -deadband) {
      armedBelow = true;
    } else if (armedBelow && v > deadband) {
      risingCrossings++;
      armedBelow = false;
    }
  }

  float durationSeconds = (n - 1) * 0.05f; // 20 Hz stored samples
  if (durationSeconds <= 0.0f || risingCrossings < 2) return NAN;

  float breathsPerMinute = risingCrossings * 60.0f / durationSeconds;
  if (breathsPerMinute < 6.0f || breathsPerMinute > 35.0f) return NAN;
  return breathsPerMinute;
}

const char *activityName() {
  switch (activityState) {
    case ACT_STATIONARY: return "STATIONARY";
    case ACT_MOVING: return "MOVING";
    case ACT_WORKOUT_LIKE: return "WORKOUT_LIKE";
    default: return "UNKNOWN";
  }
}

void printFloatOrNA(float value, int decimals) {
  if (!isfinite(value)) Serial.print("NA");
  else Serial.print(value, decimals);
}

// ---------------- SENSOR PROCESSING ----------------
void processPpgSample(
  uint32_t irValue,
  bool &beatDetected
) {

  beatDetected = false;

  // Check whether the current IR reading is valid
  bool validIR =
    irValue >= IR_FINGER_THRESHOLD &&
    irValue < IR_SATURATION_LIMIT;

  // --------------------------------
  // FINGER ACQUISITION
  // --------------------------------

  if (!fingerPresent) {

    if (!validIR) {
      fingerGoodSamples = 0;
      return;
    }

    fingerGoodSamples++;

    // Require five consecutive valid samples
    if (fingerGoodSamples < FINGER_CONFIRM_SAMPLES) {
      return;
    }

    // Finger successfully detected.
    // Reset the previous PPG calculations ONCE.
    resetPpgAndRespiration();

    fingerPresent = true;

    // Start the PPG sampling clock
    ppgSampleTimeMs = millis();
    ppgClockReady = true;

  }

  // --------------------------------
  // FINGER CONTACT MAINTENANCE
  // --------------------------------

  else {

    // Advance the timestamp for every FIFO sample,
    // including temporarily invalid samples.
    if (ppgClockReady) {

      ppgSampleTimeMs += 10;

    } else {

      ppgSampleTimeMs = millis();
      ppgClockReady = true;
    }

    if (!validIR) {

      fingerBadSamples++;

      // Only remove the finger after 75 consecutive
      // invalid samples (approximately 750 ms).
      if (fingerBadSamples >= FINGER_LOSS_SAMPLES) {

        resetPpgAndRespiration();

        fingerPresent = false;
      }

      return;
    }

    // Valid reading received: cancel the invalid counter
    fingerBadSamples = 0;
  }

  // --------------------------------
  // PPG SIGNAL PROCESSING
  // --------------------------------

  float ir = (float)irValue;

  if (!ppgDcReady) {

    ppgDc = ir;

    ppgAcFiltered = 0.0f;

    ppgPrevious1 = 0.0f;
    ppgPrevious2 = 0.0f;

    ppgDcReady = true;
  }

  // Remove DC component
  ppgDc +=
    PPG_DC_ALPHA * (ir - ppgDc);

  float ppgAc = ir - ppgDc;

  // Smooth waveform
  ppgAcFiltered +=
    PPG_SMOOTH_ALPHA *
    (ppgAc - ppgAcFiltered);

  // Adaptive signal envelope
  float absoluteAc = fabs(ppgAcFiltered);

  if (absoluteAc > ppgEnvelope) {

    ppgEnvelope = absoluteAc;

  } else {

    ppgEnvelope *= PPG_ENVELOPE_DECAY;
  }

// --------------------------------
// SPARKFUN PBA HEARTBEAT DETECTION
// --------------------------------

// Feed the original IR sample to the
// SparkFun heartbeat detection algorithm.

if (checkForBeat((int32_t)irValue)) {

  uint32_t peakTimeMs = ppgSampleTimeMs;

  beatDetected = recordBeat(peakTimeMs);
}

  // --------------------------------
  // RESPIRATION SIGNAL EXTRACTION
  // --------------------------------

  if (!respFilterReady) {

    respLowPass = ir;
    respBaseline = ir;

    respFilterReady = true;
  }

  respLowPass +=
    RESP_LP_ALPHA * (ir - respLowPass);

  respBaseline +=
    RESP_BASE_ALPHA *
    (respLowPass - respBaseline);

  float respBand =
    respLowPass - respBaseline;

  // Reduce sampling from 100 Hz to 20 Hz
  respDecimator++;

  if (respDecimator >= RESP_DECIMATION) {

    respDecimator = 0;

    storeRespSample(respBand);
  }
}

void addMpuSample(int16_t xA, int16_t yA, int16_t zA,
                  int16_t xG, int16_t yG, int16_t zG) {
  // MPU6500 configured at +/-2 g and +/-250 deg/s.
  const float ax = xA / 16384.0f;
  const float ay = yA / 16384.0f;
  const float az = zA / 16384.0f;
  const float gx = xG / 131.0f;
  const float gy = yG / 131.0f;
  const float gz = zG / 131.0f;

  float accMagnitude = sqrtf(ax * ax + ay * ay + az * az);
  float dynamicAcceleration = accMagnitude - 1.0f; // Remove gravity magnitude.

  accXSum += ax;
  accYSum += ay;
  accZSum += az;
  dynamicAccelSqSum += (double)dynamicAcceleration * dynamicAcceleration;
  gyroVectorSqSum += (double)gx * gx + (double)gy * gy + (double)gz * gz;
  mpuSamples++;
}

void updateActivity(float dynamicRms, float gyroRms) {
  lastAccelDynamicRms = dynamicRms;
  lastGyroRms = gyroRms;

  if (!isfinite(dynamicRms) || !isfinite(gyroRms)) {
    activityState = ACT_UNKNOWN;
  } else if (dynamicRms >= WORKOUT_ACCEL_RMS_G && gyroRms >= WORKOUT_GYRO_RMS_DPS) {
    activityState = ACT_WORKOUT_LIKE;
  } else if (dynamicRms >= MOVE_ACCEL_RMS_G || gyroRms >= MOVE_GYRO_RMS_DPS) {
    activityState = ACT_MOVING;
  } else {
    activityState = ACT_STATIONARY;
  }
}

// ---------------- EDA PROCESSING ----------------
void processGsrSample() {
  int raw = analogRead(GSR_PIN);
  gsrSum += raw;
  gsrSamples++;

  // Calibrate only from samples that look like skin contact.
  if (!edaCalibrated && raw <= GSR_CONTACT_RAW_MAX) {
    gsrCalibrationSum += raw;
    gsrCalibrationSamples++;
  }

  if (!edaCalibrated &&
      (uint32_t)(millis() - bootTimeMs) >= EDA_CALIBRATION_MS &&
      gsrCalibrationSamples >= 300) {
    gsrBaseline = (float)(gsrCalibrationSum / gsrCalibrationSamples);
    edaCalibrated = isfinite(gsrBaseline) && gsrBaseline > 0.0f;
  }
}

const char *edaStateName(float gsrMean, float &changePercent) {
  changePercent = NAN;
  if (!edaCalibrated) return "CALIBRATING";
  if (!isfinite(gsrMean) || gsrMean > GSR_CONTACT_RAW_MAX) {
    edaHigh = false;
    return "NO_CONTACT";
  }

  if (GSR_LOWER_RAW_MEANS_HIGHER_EDA) {
    changePercent = (gsrBaseline - gsrMean) * 100.0f / gsrBaseline;
  } else {
    changePercent = (gsrMean - gsrBaseline) * 100.0f / gsrBaseline;
  }

  // Hysteresis prevents the state from rapidly switching near the threshold.
  if (!edaHigh && changePercent >= EDA_ENTER_HIGH_PERCENT) edaHigh = true;
  else if (edaHigh && changePercent <= EDA_EXIT_HIGH_PERCENT) edaHigh = false;

  return edaHigh ? "ELEVATED_AROUSAL" : "STABLE";
}

// ---------------- WIFI / HTTP UPLOAD ----------------
// EDIT THESE THREE LINES
const char *WIFI_SSID  = "YOUR_WIFI_SSID";
const char *WIFI_PASS  = "YOUR_WIFI_PASSWORD";
const char *SERVER_URL =  "http://[2409:40c0:33:7cbc:2d:78b7:5acd:6c1f]:8000/api/readings"; // laptop LAN IP

struct Reading {
  float t, hr, hrv, edaChange, resp, ax, ay, az, dyn, gyro;
  uint32_t mpuErr;
  char edaState[20];
  char activity[16];
  char flag[18];
  char ppg[20];
};

QueueHandle_t readingQueue = nullptr;

// Writes a number, or the JSON literal null when the value is NA/NaN.
void numOrNull(char *out, size_t n, float v, int decimals) {
  if (!isfinite(v)) snprintf(out, n, "null");
  else snprintf(out, n, "%.*f", decimals, v);
}

// Runs on core 0 so slow HTTP calls never disturb the 100 Hz sensor loop on core 1.
void netTask(void *) {
  HTTPClient http;
  uint32_t lastRetryMs = 0;
  http.setReuse(true);       // keep-alive: avoids a new TCP handshake every second
  http.setTimeout(2000);

  Reading r;
  for (;;) {
    if (xQueueReceive(readingQueue, &r, portMAX_DELAY) != pdTRUE) continue;
    if (WiFi.status() != WL_CONNECTED) {            // drop reading while offline
      if (millis() - lastRetryMs > 10000) {         // and retry the connection every 10 s
        lastRetryMs = millis();
        WiFi.reconnect();
      }
      continue;
    }

    char t[12], hr[12], hrv[12], edaC[12], resp[12], ax[12], ay[12], az[12], dyn[12], gyro[12];
    numOrNull(t, sizeof(t), r.t, 1);
    numOrNull(hr, sizeof(hr), r.hr, 1);
    numOrNull(hrv, sizeof(hrv), r.hrv, 1);
    numOrNull(edaC, sizeof(edaC), r.edaChange, 1);
    numOrNull(resp, sizeof(resp), r.resp, 1);
    numOrNull(ax, sizeof(ax), r.ax, 3);
    numOrNull(ay, sizeof(ay), r.ay, 3);
    numOrNull(az, sizeof(az), r.az, 3);
    numOrNull(dyn, sizeof(dyn), r.dyn, 3);
    numOrNull(gyro, sizeof(gyro), r.gyro, 1);

    char body[512];
    snprintf(body, sizeof(body),
      "{\"t\":%s,\"hr_bpm\":%s,\"hrv_rmssd_ms\":%s,"
      "\"eda_state\":\"%s\",\"eda_change_pct\":%s,"
      "\"resp_breaths_min\":%s,"
      "\"acc_x_g\":%s,\"acc_y_g\":%s,\"acc_z_g\":%s,"
      "\"dyn_g\":%s,\"gyro_rms_dps\":%s,"
      "\"activity\":\"%s\",\"flag\":\"%s\",\"ppg_status\":\"%s\",\"mpu_err\":%lu}",
      t, hr, hrv, r.edaState, edaC, resp, ax, ay, az, dyn, gyro,
      r.activity, r.flag, r.ppg, (unsigned long)r.mpuErr);

    if (http.begin(SERVER_URL)) {
      http.addHeader("Content-Type", "application/json");
      int code = http.POST(body);
      if (code < 200 || code >= 300) {
        Serial.printf("[net] POST failed: %d\n", code);
      }
      http.end();
    }
  }
}

// Runs once at boot: tries to join Wi-Fi (up to timeoutMs) and reports the result.
bool testWiFiAtStartup(uint32_t timeoutMs = 15000) {
  Serial.println();
  Serial.print("[WiFi] Connecting to \"");
  Serial.print(WIFI_SSID);
  Serial.print("\" ");

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeoutMs) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("[WiFi] CONNECTED");
    Serial.print("[WiFi] ESP32 IP : "); Serial.println(WiFi.localIP());
    Serial.print("[WiFi] Signal   : "); Serial.print(WiFi.RSSI()); Serial.println(" dBm");
    Serial.print("[WiFi] Server   : "); Serial.println(SERVER_URL);
    return true;
  }

  Serial.println("[WiFi] NOT CONNECTED - check SSID/password, 2.4 GHz band, and range.");
  switch (WiFi.status()) {
    case WL_NO_SSID_AVAIL:   Serial.println("[WiFi] Reason: network name not found"); break;
    case WL_CONNECT_FAILED:  Serial.println("[WiFi] Reason: wrong password / auth failed"); break;
    default:                 Serial.println("[WiFi] Reason: timed out"); break;
  }
  Serial.println("[WiFi] Continuing without Wi-Fi; will keep retrying in the background.");
  return false;
}

void startNetwork() {
  readingQueue = xQueueCreate(10, sizeof(Reading));
  xTaskCreatePinnedToCore(netTask, "net", 8192, nullptr, 1, nullptr, 0);
}

// ---------------- OUTPUT ----------------
void printDashboard(uint32_t nowMs, float gsrMean, float xMean,
                    float yMean, float zMean, float dynamicRms,
                    float gyroRms, uint32_t currentMpuErrors) {
  float hr = currentHeartRate();
  float hrv = currentHRV_RMSSD();
  float edaChange = NAN;
  const char *edaLabel = edaStateName(gsrMean, edaChange);

  if (activityState == ACT_STATIONARY && isfinite(hr) &&
      hr >= (PERSONAL_RESTING_HR_BPM + REST_HR_RISE_BPM)) {
    highHrAtRestSeconds++;
  } else {
    highHrAtRestSeconds = 0;
  }
  highHrAtRest = highHrAtRestSeconds >= REST_HR_SUSTAIN_SECONDS;

  // ---- queue this second's values for the Wi-Fi task (non-blocking) ----
  if (readingQueue) {
    Reading r;
    r.t = nowMs / 1000.0f;
    r.hr = hr;
    r.hrv = hrv;
    r.edaChange = edaChange;
    r.resp = latestRespRate;
    r.ax = xMean; r.ay = yMean; r.az = zMean;
    r.dyn = dynamicRms;
    r.gyro = gyroRms;
    r.mpuErr = currentMpuErrors;
    strlcpy(r.edaState, edaLabel, sizeof(r.edaState));
    strlcpy(r.activity, activityName(), sizeof(r.activity));
    strlcpy(r.flag, highHrAtRest ? "HIGH_HR_AT_REST" : "NONE", sizeof(r.flag));
    strlcpy(r.ppg, !fingerPresent ? "NO_FINGER_OR_SIGNAL" : (isfinite(hr) ? "OK" : "ACQUIRING"),
            sizeof(r.ppg));
    xQueueSend(readingQueue, &r, 0);   // 0 = never block the sensor loop
  }

  Serial.print("t=");
  Serial.print(nowMs / 1000.0f, 1);
  Serial.print("s | HR=");
  printFloatOrNA(hr, 1);
  Serial.print(" bpm | HRV_RMSSD=");
  printFloatOrNA(hrv, 1);
  Serial.print(" ms | EDA=");
  Serial.print(edaLabel);
  Serial.print(" (");
  printFloatOrNA(edaChange, 1);
  Serial.print("% vs baseline) | RESP_PPG=");
  printFloatOrNA(latestRespRate, 1);
  Serial.print(" breaths/min | ACC[g]=");
  printFloatOrNA(xMean, 3);
  Serial.print(",");
  printFloatOrNA(yMean, 3);
  Serial.print(",");
  printFloatOrNA(zMean, 3);
  Serial.print(" | DYN=");
  printFloatOrNA(dynamicRms, 3);
  Serial.print("g | GYRO_RMS=");
  printFloatOrNA(gyroRms, 1);
  Serial.print(" deg/s | ACTIVITY=");
  Serial.print(activityName());
  Serial.print(" | FLAG=");
  Serial.print(highHrAtRest ? "HIGH_HR_AT_REST" : "NONE");
  Serial.print(" | PPG=");
  if (!fingerPresent) Serial.print("NO_FINGER_OR_SIGNAL");
  else if (isfinite(hr)) Serial.print("OK");
  else Serial.print("ACQUIRING");
  Serial.print(" | MPU_ERR=");
  Serial.println(currentMpuErrors);
}

void outputPpgPlot(bool beatDetected) {
  float normalizedPpg = 0.0f;
  if (fingerPresent && ppgEnvelope > 1.0f) {
    normalizedPpg = (ppgAcFiltered / ppgEnvelope) * 100.0f;
  }
  float hr = currentHeartRate();

  Serial.print("PPG:");
  Serial.print(normalizedPpg, 2);
  Serial.print("\tHR_BPM:");
  Serial.print(isfinite(hr) ? hr : 0.0f, 1);
  Serial.print("\tBEAT:");
  Serial.println(beatDetected ? 150 : 0);
}

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(115200);
  delay(1000);
  testWiFiAtStartup();       // Wi-Fi check first, so it doesn't eat the 10 s EDA calibration window
  bootTimeMs = millis();     // calibration clock starts AFTER the Wi-Fi test
  startNetwork();

  analogReadResolution(12);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000); // Keep the known-stable I2C speed for both sensors.

  mpuReady = initMPU();
  maxReady = particleSensor.begin(Wire, I2C_SPEED_STANDARD);

  if (maxReady) {
    particleSensor.setup(0x1F, 1, 2, 100, 411, 4096);
    particleSensor.setPulseAmplitudeRed(0x00); // Red channel not needed for HR here.
    particleSensor.setPulseAmplitudeIR(0x1F);
  }

#if !OUTPUT_PPG_PLOT
  Serial.println();
  Serial.println("WEARABLE ANXIETY MONITOR | 1-second dashboard");
  Serial.println("Keep GSR electrodes on skin during initial baseline calibration.");
  Serial.print("MPU6500: "); Serial.println(mpuReady ? "READY" : "NOT FOUND");
  Serial.print("MAX30102: "); Serial.println(maxReady ? "READY" : "NOT FOUND");
  Serial.println("HRV is rolling RMSSD; respiration is an experimental PPG estimate.");
#endif

  resetMpuWindow();
  lastMpuSampleUs = micros();
  lastReportMs = millis();
  lastRespReportMs = millis();
}

// ---------------- MAIN LOOP ----------------
void loop() {
  // 1) Drain every available MAX30102 FIFO sample. Do not use getIR() here:
  //    getFIFOIR() reads the next queued sample without waiting for another one.
  if (maxReady) {
    particleSensor.check();

    while (particleSensor.available()) {
      uint32_t irValue = particleSensor.getFIFOIR();
      bool beatDetected = false;
      processPpgSample(irValue, beatDetected);
#if OUTPUT_PPG_PLOT
      outputPpgPlot(beatDetected);
#endif
      particleSensor.nextSample();
    }
  }

  // 2) Sample GSR and MPU approximately every 10 ms.
  uint32_t nowUs = micros();
  if ((uint32_t)(nowUs - lastMpuSampleUs) >= MPU_PERIOD_US) {
    lastMpuSampleUs = nowUs;
    processGsrSample();

    if (mpuReady) {
      int16_t xA, yA, zA, xG, yG, zG;
      if (readMPU(xA, yA, zA, xG, yG, zG)) {
        addMpuSample(xA, yA, zA, xG, yG, zG);
      } else {
        mpuErrors++;
      }
    }
  }

  uint32_t nowMs = millis();

  // 3) Update the respiration estimate every five seconds.
  if ((uint32_t)(nowMs - lastRespReportMs) >= RESP_REPORT_PERIOD_MS) {
    lastRespReportMs = nowMs;
    bool sufficientlyStill = isfinite(lastAccelDynamicRms) &&
                             isfinite(lastGyroRms) &&
                             lastAccelDynamicRms < 0.08f &&
                             lastGyroRms < 8.0f;
    if (fingerPresent && sufficientlyStill) {
      latestRespRate = estimateRespirationRate();
    } else {
      latestRespRate = NAN; // Reject respiration estimate during motion/poor contact.
    }
  }

  // 4) Print the physiological and activity dashboard once per second.
  if ((uint32_t)(nowMs - lastReportMs) >= REPORT_PERIOD_MS) {
    lastReportMs = nowMs;

#if !OUTPUT_PPG_PLOT
    float gsrMean = gsrSamples ? (float)(gsrSum / gsrSamples) : NAN;
    float xMean = NAN, yMean = NAN, zMean = NAN;
    float dynamicRms = NAN, gyroRms = NAN;

    if (mpuSamples > 0) {
      xMean = (float)(accXSum / mpuSamples);
      yMean = (float)(accYSum / mpuSamples);
      zMean = (float)(accZSum / mpuSamples);
      dynamicRms = sqrt((float)(dynamicAccelSqSum / mpuSamples));
      gyroRms = sqrt((float)(gyroVectorSqSum / mpuSamples));
    }

    updateActivity(dynamicRms, gyroRms);
    printDashboard(nowMs, gsrMean, xMean, yMean, zMean,
                   dynamicRms, gyroRms, mpuErrors);
#endif

    gsrSum = 0.0;
    gsrSamples = 0;
    resetMpuWindow();
  }
}