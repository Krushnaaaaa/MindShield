
import { useEffect, useState } from "react";
import {
  Activity,
  HeartPulse,
  Wind,
  Waves,
  Footprints,
  Wifi,
  WifiOff,
  RefreshCw,
  Clock,
  AlertTriangle,
} from "lucide-react";

import "./App.css";

// Change this if your backend is running on another computer.
// You can also set VITE_API_URL in your frontend .env file.
const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const POLL_INTERVAL = 1000;
const CONNECTION_TIMEOUT = 5000;

function getValue(reading, primary, alternative) {
  return reading?.[primary] ?? reading?.[alternative] ?? null;
}

function formatValue(value, decimals = 1) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }

  const number = Number(value);

  return Number.isFinite(number) ? number.toFixed(decimals) : "--";
}

function MetricCard({
  title,
  value,
  unit,
  icon: Icon,
  description,
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <div>
          <p className="metric-title">{title}</p>
          <h2 className="metric-value">
            {value}
            <span className="metric-unit">{unit}</span>
          </h2>
        </div>

        <div className="metric-icon">
          <Icon size={23} />
        </div>
      </div>

      <p className="metric-description">{description}</p>
    </div>
  );
}

function App() {
  const [reading, setReading] = useState(null);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchLatestReading = async () => {
    const controller = new AbortController();

    const timeout = setTimeout(
      () => controller.abort(),
      CONNECTION_TIMEOUT
    );

    try {
      const response = await fetch(
        `${API_URL}/api/readings/latest`,
        {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
          signal: controller.signal,
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const result = await response.json();

      setConnected(true);
      setLoading(false);

      if (result.success && result.reading) {
        setReading(result.reading);
        setLastUpdated(new Date());
        setError("");
      } else {
        setReading(null);
        setError(result.message || "Waiting for ESP32 sensor data...");
      }
    } catch (err) {
      setConnected(false);
      setLoading(false);

      if (err.name === "AbortError") {
        setError("Backend connection timed out.");
      } else {
        setError(err.message || "Cannot connect to backend.");
      }
    } finally {
      clearTimeout(timeout);
    }
  };

  useEffect(() => {
    fetchLatestReading();

    const interval = setInterval(
      fetchLatestReading,
      POLL_INTERVAL
    );

    return () => clearInterval(interval);
  }, []);

  const hr = getValue(reading, "hr_bpm", "hr");
  const hrv = getValue(reading, "hrv_rmssd_ms", "hrv");
  const eda = getValue(reading, "eda_change_pct", "eda");
  const respiration = getValue(
    reading,
    "resp_breaths_min",
    "resp"
  );

  const accX = getValue(reading, "acc_x_g", "acc_x");
  const accY = getValue(reading, "acc_y_g", "acc_y");
  const accZ = getValue(reading, "acc_z_g", "acc_z");

  const activity = reading?.activity ?? "--";
  const edaState = reading?.eda_state ?? "--";
  const dynamicG = reading?.dyn_g ?? null;
  const gyroRms = reading?.gyro_rms_dps ?? null;
  const ppgStatus = reading?.ppg_status ?? "--";
  const sensorFlag = reading?.flag ?? "--";

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={28} />
          </div>

          <div>
            <h1>MindShield</h1>
            <p>AI-Based Physiological Monitoring System</p>
          </div>
        </div>

        <div
          className={`connection ${
            connected ? "online" : "offline"
          }`}
        >
          {connected ? (
            <Wifi size={18} />
          ) : (
            <WifiOff size={18} />
          )}

          <span>
            {connected ? "BACKEND CONNECTED" : "DISCONNECTED"}
          </span>
        </div>
      </header>

      <main className="main">
        <section className="welcome">
          <div>
            <p className="eyebrow">LIVE HARDWARE DASHBOARD</p>
            <h2>Sensor Monitoring</h2>
            <p className="subtitle">
              Real-time readings received from your ESP32 hardware.
            </p>
          </div>

          <button
            className="refresh-button"
            onClick={fetchLatestReading}
          >
            <RefreshCw size={16} />
            Refresh
          </button>
        </section>

        <section className="status-panel">
          <div className="status-left">
            <span
              className={`status-dot ${
                reading ? "active" : "inactive"
              }`}
            />

            <div>
              <strong>
                {reading
                  ? "Sensor data received"
                  : connected
                  ? "Waiting for sensor readings"
                  : "Backend unavailable"}
              </strong>

              <p>
                {reading
                  ? "Displaying the latest ESP32 reading."
                  : error || "No sensor data available."}
              </p>
            </div>
          </div>

          <div className="status-right">
            <span>DATA SOURCE</span>
            <strong>ESP32 HARDWARE</strong>
          </div>
        </section>

        {loading && (
          <div className="message-panel">
            Connecting to MindShield backend...
          </div>
        )}

        {!loading && !reading && (
          <div className="message-panel warning">
            <AlertTriangle size={20} />
            <div>
              <strong>No sensor reading available</strong>
              <p>{error}</p>
              <p>
                Check that your ESP32 is uploading data to
                POST /api/readings.
              </p>
            </div>
          </div>
        )}

        <section className="section-heading">
          <div>
            <h3>Physiological Measurements</h3>
            <p>Values from the latest received hardware reading</p>
          </div>
        </section>

        <section className="metrics-grid">
          <MetricCard
            title="Heart Rate"
            value={formatValue(hr, 0)}
            unit="BPM"
            icon={HeartPulse}
            description="Pulse rate from MAX30102"
          />

          <MetricCard
            title="Heart Rate Variability"
            value={formatValue(hrv, 1)}
            unit="ms"
            icon={Activity}
            description="RMSSD value reported by ESP32"
          />

          <MetricCard
            title="EDA Change"
            value={formatValue(eda, 1)}
            unit="%"
            icon={Waves}
            description={`EDA state: ${edaState}`}
          />

          <MetricCard
            title="Respiration"
            value={formatValue(respiration, 0)}
            unit="BPM"
            icon={Wind}
            description="Breaths per minute reported by ESP32"
          />
        </section>

        <section className="section-heading">
          <div>
            <h3>Movement & Sensor Information</h3>
            <p>Accelerometer, gyroscope and sensor status</p>
          </div>
        </section>

        <section className="details-grid">
          <div className="detail-card">
            <div className="detail-heading">
              <Footprints size={20} />
              <h3>Movement Analysis</h3>
            </div>

            <div className="activity-value">
              {String(activity).toUpperCase()}
            </div>

            <div className="detail-row">
              <span>Dynamic acceleration</span>
              <strong>{formatValue(dynamicG, 3)} g</strong>
            </div>

            <div className="detail-row">
              <span>Gyroscope RMS</span>
              <strong>{formatValue(gyroRms, 2)} °/s</strong>
            </div>
          </div>

          <div className="detail-card">
            <div className="detail-heading">
              <Activity size={20} />
              <h3>Accelerometer</h3>
            </div>

            <div className="detail-row">
              <span>X axis</span>
              <strong>{formatValue(accX, 3)} g</strong>
            </div>

            <div className="detail-row">
              <span>Y axis</span>
              <strong>{formatValue(accY, 3)} g</strong>
            </div>

            <div className="detail-row">
              <span>Z axis</span>
              <strong>{formatValue(accZ, 3)} g</strong>
            </div>
          </div>

          <div className="detail-card">
            <div className="detail-heading">
              <Wifi size={20} />
              <h3>Hardware Status</h3>
            </div>

            <div className="detail-row">
              <span>PPG status</span>
              <strong>{String(ppgStatus)}</strong>
            </div>

            <div className="detail-row">
              <span>Sensor flag</span>
              <strong>{String(sensorFlag)}</strong>
            </div>

            <div className="detail-row">
              <span>EDA state</span>
              <strong>{String(edaState)}</strong>
            </div>
          </div>
        </section>

        <section className="last-updated">
          <Clock size={16} />

          <span>
            Last reading received:{" "}
            {lastUpdated
              ? lastUpdated.toLocaleTimeString()
              : "Not available"}
          </span>
        </section>

        <footer className="footer">
          <p>
            MindShield V2.0 · Experimental monitoring prototype
          </p>
          <p>
            Sensor measurements are not a medical diagnosis.
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
