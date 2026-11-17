
import { useCallback, useEffect, useMemo, useState } from "react";
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
  ShieldAlert,
  MapPin,
  Send,
  CheckCircle2,
  XCircle,
  X,
  Download,
  History,
} from "lucide-react";

import "./App.css";

const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const POLL_INTERVAL = 1000;
const HISTORY_POLL_INTERVAL = 10000;
const CONNECTION_TIMEOUT = 5000;
const MAX_GRAPH_POINTS = 60;

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

function formatDate(value) {
  if (!value) return "--";

  const date = new Date(value);

  return Number.isNaN(date.getTime())
    ? "--"
    : date.toLocaleString();
}

// --------------------------------------------------
// METRIC CARD
// --------------------------------------------------

function MetricCard({ title, value, unit, icon: Icon, description }) {
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

// --------------------------------------------------
// REUSABLE LIVE PHYSIOLOGICAL LINE GRAPH
// --------------------------------------------------

function LineChart({
  title,
  unit,
  data,
  color = "#6366f1",
  decimals = 1,
}) {
  const width = 600;
  const height = 190;
  const padding = 30;

  const values = data
    .map((point) => Number(point.value))
    .filter(Number.isFinite);

  const hasData = values.length > 0;

  const minimum = hasData ? Math.min(...values) : 0;
  const maximum = hasData ? Math.max(...values) : 1;

  const range = maximum - minimum || 1;

  const points = data
    .map((point, index) => {
      const value = Number(point.value);

      if (!Number.isFinite(value)) return null;

      const x =
        data.length <= 1
          ? width / 2
          : padding +
            (index / (data.length - 1)) *
              (width - padding * 2);

      const y =
        height -
        padding -
        ((value - minimum) / range) *
          (height - padding * 2);

      return `${x},${y}`;
    })
    .filter(Boolean);

  const latestValue = hasData
    ? values[values.length - 1]
    : null;

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div>
          <h4>{title}</h4>
          <p>Live sensor trend · Last {MAX_GRAPH_POINTS} readings</p>
        </div>

        <div className="chart-current-value">
          {formatValue(latestValue, decimals)}
          <span>{unit}</span>
        </div>
      </div>

      <div className="chart-container">
        {hasData && (
          <svg
            className="line-chart"
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label={`${title} live graph`}
            preserveAspectRatio="none"
          >
            {[0, 1, 2, 3].map((line) => {
              const y =
                padding +
                (line / 3) * (height - padding * 2);

              return (
                <line
                  key={line}
                  x1={padding}
                  y1={y}
                  x2={width - padding}
                  y2={y}
                  stroke="#e5e7eb"
                  strokeDasharray="5 5"
                />
              );
            })}

            <polyline
              points={points.join(" ")}
              fill="none"
              stroke={color}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </div>

      <div className="chart-footer">
        <span>
          Min: {formatValue(hasData ? minimum : null, decimals)}
        </span>

        <span>
          Max: {formatValue(hasData ? maximum : null, decimals)}
        </span>
      </div>
    </div>
  );
}

// --------------------------------------------------
// BREATHING VISUALIZATION
// --------------------------------------------------

function BreathingVisual({ respiration }) {
  const rate = Number(respiration);

  const validRate =
    Number.isFinite(rate) && rate > 0
      ? Math.min(Math.max(rate, 4), 40)
      : 12;

  const cycleDuration = 60 / validRate;

  return (
    <div className="breathing-card">
      <div className="breathing-heading">
        <Wind size={21} />

        <div>
          <h4>Breathing Activity</h4>
          <p>Visual pace based on reported respiration</p>
        </div>
      </div>

      <div className="breathing-display">
        <div
          className="breathing-circle"
          style={{
            "--breath-duration": `${cycleDuration}s`,
          }}
        >
          <Wind size={35} />
          <strong>{formatValue(respiration, 0)}</strong>
          <span>breaths/min</span>
        </div>
      </div>

      <p className="breathing-note">
        Experimental visualization, not a measured breathing waveform.
      </p>
    </div>
  );
}

// --------------------------------------------------
// MAIN APPLICATION
// --------------------------------------------------

function App() {
  const [reading, setReading] = useState(null);
  const [connected, setConnected] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Physiological graph data
  const [graphData, setGraphData] = useState([]);

  // Patient history
  const [history, setHistory] = useState([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  // SOS history
  const [sosHistory, setSosHistory] = useState([]);

  // Manual SOS
  const [showSosConfirm, setShowSosConfirm] = useState(false);
  const [sosSending, setSosSending] = useState(false);
  const [sosResult, setSosResult] = useState(null);
  const [sosError, setSosError] = useState("");

  // ------------------------------------------------
  // FETCH LATEST ESP32 READING
  // ------------------------------------------------

  const fetchLatestReading = useCallback(async () => {
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
          headers: { Accept: "application/json" },
          signal: controller.signal,
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const result = await response.json();

      setConnected(true);

      if (result.success && result.reading) {
        const latest = result.reading;

        setReading(latest);
        setLastUpdated(new Date());

        setGraphData((previous) => {
          const latestId =
            latest.id ?? latest.t ?? latest.received_at;

          const previousId =
            previous.length > 0
              ? previous[previous.length - 1].id
              : null;

          if (latestId === previousId) {
            return previous;
          }

          return [
            ...previous,
            {
              id: latestId,
              time: latest.received_at ?? new Date().toISOString(),
              hr: getValue(latest, "hr_bpm", "hr"),
              hrv: getValue(latest, "hrv_rmssd_ms", "hrv"),
              eda: getValue(latest, "eda_change_pct", "eda"),
              respiration: getValue(
                latest,
                "resp_breaths_min",
                "resp"
              ),
            },
          ].slice(-MAX_GRAPH_POINTS);
        });
      } else {
        setReading(null);
      }
    } catch (err) {
      setConnected(false);
    } finally {
      clearTimeout(timeout);
    }
  }, []);

  // ------------------------------------------------
  // FETCH PATIENT SENSOR HISTORY
  // ------------------------------------------------

  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const params = new URLSearchParams({
        limit: "300",
        offset: "0",
      });

      if (startDate) {
        params.set("start_date", startDate);
      }

      if (endDate) {
        params.set("end_date", endDate);
      }

      const response = await fetch(
        `${API_URL}/api/readings/history?${params.toString()}`,
        {
          headers: { Accept: "application/json" },
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `History request failed: HTTP ${response.status}`
        );
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error("Could not load patient history.");
      }

      setHistory(result.readings || []);
      setHistoryTotal(result.total || 0);
    } catch (err) {
      setHistoryError(
        err.message || "Could not load patient history."
      );
    } finally {
      setHistoryLoading(false);
    }
  }, [startDate, endDate]);

  // ------------------------------------------------
  // FETCH SOS HISTORY
  // ------------------------------------------------

  const fetchSosHistory = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_URL}/api/sos/history?limit=100`,
        {
          headers: { Accept: "application/json" },
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error("Could not load SOS history.");
      }

      const result = await response.json();

      if (result.success) {
        setSosHistory(result.events || []);
      }
    } catch (err) {
      console.error("SOS history error:", err);
    }
  }, []);

  // ------------------------------------------------
  // POLLING
  // ------------------------------------------------

  useEffect(() => {
    fetchLatestReading();

    const interval = setInterval(
      fetchLatestReading,
      POLL_INTERVAL
    );

    return () => clearInterval(interval);
  }, [fetchLatestReading]);

  useEffect(() => {
    fetchHistory();
    fetchSosHistory();

    const interval = setInterval(() => {
      fetchHistory();
      fetchSosHistory();
    }, HISTORY_POLL_INTERVAL);

    return () => clearInterval(interval);
  }, [fetchHistory, fetchSosHistory]);

  // ------------------------------------------------
  // CURRENT SENSOR VALUES
  // ------------------------------------------------

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

  // ------------------------------------------------
  // PHYSIOLOGICAL GRAPH SERIES
  // ------------------------------------------------

  const heartRateGraph = useMemo(
    () =>
      graphData.map((point) => ({
        time: point.time,
        value: point.hr,
      })),
    [graphData]
  );

  const hrvGraph = useMemo(
    () =>
      graphData.map((point) => ({
        time: point.time,
        value: point.hrv,
      })),
    [graphData]
  );

  const edaGraph = useMemo(
    () =>
      graphData.map((point) => ({
        time: point.time,
        value: point.eda,
      })),
    [graphData]
  );

  const respirationGraph = useMemo(
    () =>
      graphData.map((point) => ({
        time: point.time,
        value: point.respiration,
      })),
    [graphData]
  );

  // ------------------------------------------------
  // MANUAL SOS
  // ------------------------------------------------

  const startManualSos = () => {
    setSosError("");
    setSosResult(null);
    setShowSosConfirm(true);
  };

  const sendManualSos = () => {
    if (sosSending) return;

    if (!navigator.geolocation) {
      setSosError(
        "This browser does not support location access. Open MindShield in a supported browser and allow location permission."
      );

      setShowSosConfirm(false);
      return;
    }

    setSosSending(true);
    setSosError("");
    setSosResult(null);

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;

        const payload = {
          patient_name: "MindShield User",
          reason: "Manual emergency SOS triggered by user",
          latitude,
          longitude,
          stress_mindset: "MANUAL SOS",
          ai_probability: 0,
          heart_rate: hr === null ? null : Number(hr),
          hrv: hrv === null ? null : Number(hrv),
          respiration:
            respiration === null ? null : Number(respiration),
          eda: eda === null ? null : Number(eda),
          notes: reading
            ? "Manual SOS triggered from MindShield dashboard. Latest ESP32 readings attached where available."
            : "Manual SOS triggered from MindShield dashboard. No current ESP32 reading was available.",
          warning_seconds: null,
        };

        try {
          const response = await fetch(`${API_URL}/sos`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
            body: JSON.stringify(payload),
          });

          const result = await response.json().catch(() => ({}));

          if (!response.ok) {
            throw new Error(
              result.detail ||
                result.message ||
                `SOS request failed with HTTP ${response.status}`
            );
          }

          setSosResult({
            ...result,
            latitude,
            longitude,
            locationLink:
              `https://www.openstreetmap.org/?mlat=${latitude}&mlon=${longitude}#map=18/${latitude}/${longitude}`,
          });

          setShowSosConfirm(false);
          fetchSosHistory();
        } catch (err) {
          setSosError(
            err.message ||
              "Could not send the SOS request. Check the backend connection."
          );
        } finally {
          setSosSending(false);
        }
      },

      (geoError) => {
        let message = "Unable to get your current location.";

        if (geoError.code === 1) {
          message =
            "Location permission was denied. Allow location access in your browser and try again.";
        } else if (geoError.code === 2) {
          message =
            "Your current location is unavailable. Check device location services and try again.";
        } else if (geoError.code === 3) {
          message =
            "Location request timed out. Please try again.";
        }

        setSosError(message);
        setSosSending(false);
      },

      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 10000,
      }
    );
  };

  // ------------------------------------------------
  // EXPORT HISTORY TO CSV
  // ------------------------------------------------

  const exportHistoryCSV = () => {
    if (!history.length) {
      setHistoryError("There are no history records to export.");
      return;
    }

    const columns = [
      "id",
      "received_at",
      "t",
      "hr_bpm",
      "hrv_rmssd_ms",
      "eda_state",
      "eda_change_pct",
      "resp_breaths_min",
      "acc_x_g",
      "acc_y_g",
      "acc_z_g",
      "dyn_g",
      "gyro_rms_dps",
      "activity",
      "flag",
      "ppg_status",
      "mpu_err",
    ];

    const escapeCSV = (value) => {
      const text = value ?? "";
      return `"${String(text).replace(/"/g, '""')}"`;
    };

    const rows = [
      columns.join(","),
      ...history.map((item) =>
        columns
          .map((column) => escapeCSV(item[column]))
          .join(",")
      ),
    ];

    const csv = rows.join("\n");

    const blob = new Blob([csv], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "mindshield_patient_history.csv";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  // ------------------------------------------------
  // RENDER
  // ------------------------------------------------

  return (
    <div className="app">

      {/* HEADER */}

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

        {/* DASHBOARD INTRO */}

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
            onClick={() => {
              fetchLatestReading();
              fetchHistory();
              fetchSosHistory();
            }}
            type="button"
          >
            <RefreshCw size={16} />
            Refresh
          </button>
        </section>

        {/* SIMPLE MANUAL SOS BUTTON */}

        <div style={{ marginBottom: "28px" }}>
          <button
            type="button"
            onClick={startManualSos}
            disabled={sosSending}
            style={{
              backgroundColor: "#dc2626",
              color: "#ffffff",
              border: "none",
              borderRadius: "8px",
              padding: "13px 22px",
              fontSize: "14px",
              fontWeight: "700",
              letterSpacing: "0.4px",
              cursor: sosSending ? "not-allowed" : "pointer",
              opacity: sosSending ? 0.7 : 1,
              display: "inline-flex",
              alignItems: "center",
              gap: "9px",
            }}
          >
            <ShieldAlert size={19} />
            {sosSending ? "SENDING SOS..." : "TRIGGER MANUAL SOS"}
          </button>
        </div>

        {/* SOS CONFIRMATION */}

        {showSosConfirm && (
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="sos-confirm-title"
            className="sos-confirm-panel"
          >
            <div className="sos-confirm-header">
              <h3 id="sos-confirm-title">
                Confirm emergency SOS
              </h3>

              <button
                type="button"
                aria-label="Close confirmation"
                onClick={() => setShowSosConfirm(false)}
                disabled={sosSending}
                className="icon-button"
              >
                <X size={20} />
              </button>
            </div>

            <p>
              This will send a manual SOS request to your
              MindShield backend. Your browser will ask for
              location permission. Continue only if you want
              to notify your configured emergency contact.
            </p>

            <div className="sos-actions">
              <button
                type="button"
                onClick={sendManualSos}
                disabled={sosSending}
                className="confirm-sos-button"
              >
                <Send size={17} />
                {sosSending ? "Sending..." : "Yes, send SOS"}
              </button>

              <button
                type="button"
                onClick={() => setShowSosConfirm(false)}
                disabled={sosSending}
                className="cancel-button"
              >
                Cancel
              </button>
            </div>
          </section>
        )}

        {sosSending && (
          <div role="status" className="sos-progress">
            <Send size={17} />
            Requesting location and sending SOS to the backend...
          </div>
        )}

        {sosError && (
          <div role="alert" className="sos-error">
            <XCircle size={19} />
            <span>{sosError}</span>
          </div>
        )}

        {sosResult && (
          <div role="status" className="sos-success">
            <div className="sos-success-title">
              <CheckCircle2 size={20} />
              SOS request accepted by MindShield backend
            </div>

            <p>
              This confirms the backend accepted the request.
              Check the backend log and WhatsApp recipient to
              verify Twilio delivery.
            </p>

            {sosResult.sms?.status && (
              <p>
                WhatsApp status:{" "}
                <strong>{sosResult.sms.status}</strong>
              </p>
            )}

            <a
              href={sosResult.locationLink}
              target="_blank"
              rel="noreferrer"
            >
              <MapPin size={16} />
              Open reported location
            </a>
          </div>
        )}

        {/* PHYSIOLOGICAL MEASUREMENTS */}

        <section className="section-heading">
          <div>
            <h3>Physiological Measurements</h3>
            <p>
              Live values from your MAX30102 and EDA sensor
            </p>
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

        {/* LIVE PHYSIOLOGICAL GRAPHS */}

        <section className="section-heading">
          <div>
            <h3>Live Physiological Graphs</h3>
            <p>
              Continuously updated trends from incoming sensor readings
            </p>
          </div>
        </section>

        <section className="charts-grid">
          <LineChart
            title="Heart Rate"
            unit="BPM"
            data={heartRateGraph}
            color="#ef4444"
            decimals={0}
          />

          <LineChart
            title="Heart Rate Variability"
            unit="ms"
            data={hrvGraph}
            color="#6366f1"
          />

          <LineChart
            title="EDA Change"
            unit="%"
            data={edaGraph}
            color="#0d9488"
          />

          <LineChart
            title="Respiration Rate"
            unit="breaths/min"
            data={respirationGraph}
            color="#0284c7"
            decimals={0}
          />
        </section>

        {/* BREATHING VISUALIZATION */}

        <section className="breathing-section">
          <BreathingVisual respiration={respiration} />
        </section>

        {/* MOVEMENT AND SENSOR INFORMATION */}

        <section className="section-heading">
          <div>
            <h3>Movement & Sensor Information</h3>
            <p>
              Accelerometer, gyroscope and hardware status
            </p>
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

        {/* PATIENT HISTORY */}

        <section className="section-heading history-heading">
          <div>
            <h3>
              <History size={21} />
              Patient Sensor History
            </h3>

            <p>
              Historical readings saved in the MindShield database
            </p>
          </div>

          <button
            type="button"
            className="export-button"
            onClick={exportHistoryCSV}
            disabled={!history.length}
          >
            <Download size={17} />
            Export CSV
          </button>
        </section>

        <section className="history-card">

          <div className="history-summary">
            <div>
              <span>Total saved readings</span>
              <strong>{historyTotal}</strong>
            </div>

            <div>
              <span>Records displayed</span>
              <strong>{history.length}</strong>
            </div>
          </div>

          <div className="history-filters">
            <label>
              Start date
              <input
                type="date"
                value={startDate}
                onChange={(event) =>
                  setStartDate(event.target.value)
                }
              />
            </label>

            <label>
              End date
              <input
                type="date"
                value={endDate}
                onChange={(event) =>
                  setEndDate(event.target.value)
                }
              />
            </label>

            <button
              type="button"
              onClick={fetchHistory}
              disabled={historyLoading}
            >
              <RefreshCw size={15} />
              {historyLoading ? "Loading..." : "Apply filter"}
            </button>

            <button
              type="button"
              className="clear-filter-button"
              onClick={() => {
                setStartDate("");
                setEndDate("");
              }}
            >
              Clear
            </button>
          </div>

          {historyError && (
            <div className="history-error">
              <AlertTriangle size={17} />
              {historyError}
            </div>
          )}

          <div className="history-table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Heart Rate</th>
                  <th>HRV</th>
                  <th>EDA</th>
                  <th>Respiration</th>
                  <th>Activity</th>
                  <th>PPG</th>
                </tr>
              </thead>

              <tbody>
                {history.length > 0 ? (
                  history.map((item) => (
                    <tr key={item.id}>
                      <td>{formatDate(item.received_at)}</td>
                      <td>{formatValue(item.hr_bpm, 0)} BPM</td>
                      <td>{formatValue(item.hrv_rmssd_ms)} ms</td>
                      <td>{formatValue(item.eda_change_pct)}%</td>
                      <td>{formatValue(item.resp_breaths_min, 0)}</td>
                      <td>{item.activity ?? "--"}</td>
                      <td>{item.ppg_status ?? "--"}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="7" className="empty-history">
                      {historyLoading
                        ? "Loading patient history..."
                        : "No saved readings found. Connect your ESP32 to start building patient history."}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <p className="history-note">
            Showing the latest 300 matching readings. Older
            records remain saved in the database.
          </p>

        </section>

        {/* SOS HISTORY */}

        <section className="section-heading">
          <div>
            <h3>
              <ShieldAlert size={21} />
              Emergency SOS History
            </h3>

            <p>
              Previously recorded manual and automatic SOS requests
            </p>
          </div>
        </section>

        <section className="sos-history-card">
          {sosHistory.length > 0 ? (
            sosHistory.map((event) => (
              <div
                className="sos-history-item"
                key={event.id}
              >
                <div className="sos-history-icon">
                  <ShieldAlert size={20} />
                </div>

                <div className="sos-history-details">
                  <strong>
                    {event.patient_name || "Patient"}
                  </strong>

                  <span>{formatDate(event.created_at)}</span>

                  <p>
                    {event.request?.reason || "SOS request"}
                  </p>

                  {event.error_message && (
                    <p className="sos-history-error">
                      {event.error_message}
                    </p>
                  )}
                </div>

                <span
                  className={`sos-history-status ${
                    event.status === "PROCESSED"
                      ? "processed"
                      : "failed"
                  }`}
                >
                  {event.status}
                </span>
              </div>
            ))
          ) : (
            <div className="empty-history">
              No SOS events have been recorded yet.
            </div>
          )}
        </section>

        {/* LAST UPDATED */}

        <section className="last-updated">
          <Clock size={16} />

          <span>
            Last reading received:{" "}
            {lastUpdated
              ? lastUpdated.toLocaleTimeString()
              : "Not available"}
          </span>
        </section>

        {/* FOOTER */}

        <footer className="footer">
          <p>
            MindShield V2.0 · Experimental monitoring prototype
          </p>

          <p>
            Sensor measurements and model outputs are not a
            medical diagnosis. In a real emergency, contact
            local emergency services as well.
          </p>
        </footer>

      </main>
    </div>
  );
}

export default App;
