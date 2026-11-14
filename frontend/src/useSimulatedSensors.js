
import { useState, useEffect, useRef } from "react";

const random = (min, max) =>
  Math.random() * (max - min) + min;

const clamp = (value, min, max) =>
  Math.min(max, Math.max(min, value));

export default function useSimulatedSensors() {

  const [readings, setReadings] = useState({
    heartRate: 76,
    hrv: 48,
    eda: 1.2,
    temperature: 33.5,
    respiration: 16,
    movement: 0.25,
    stress: "NORMAL",
  });

  const [history, setHistory] = useState([]);

  const [signals, setSignals] = useState({
    ppg: [],
    eda: [],
    accelerometer: [],
  });

  // Refs keep the signal generators running smoothly.
  const phase = useRef(0);
  const sampleNumber = useRef(0);
  const currentBPM = useRef(76);
  const edaLevel = useRef(1.2);
  const accel = useRef({ x: 0, y: 0, z: 1 });

  // ------------------------------------------
  // 1. UPDATE DASHBOARD READINGS EVERY SECOND
  // ------------------------------------------

  useEffect(() => {

    const interval = setInterval(() => {

      setReadings((previous) => {

        const heartRate = Math.round(
          clamp(previous.heartRate + random(-3, 3), 55, 130)
        );

        const hrv = Number(
          clamp(previous.hrv + random(-2, 2), 15, 100).toFixed(1)
        );

        const eda = Number(
          clamp(previous.eda + random(-0.08, 0.08), 0.1, 5).toFixed(2)
        );

        const temperature = Number(
          clamp(
            previous.temperature + random(-0.05, 0.05),
            30,
            37
          ).toFixed(1)
        );

        const respiration = Math.round(
          clamp(previous.respiration + random(-1, 1), 10, 30)
        );

        const movement = Number(
          clamp(previous.movement + random(-0.1, 0.1), 0, 2).toFixed(2)
        );

        let stress = "NORMAL";

        if (heartRate > 100 && eda > 2.5) {
          stress = "HIGH";
        } else if (heartRate > 90 || eda > 1.8) {
          stress = "MODERATE";
        }

        const updated = {
          heartRate,
          hrv,
          eda,
          temperature,
          respiration,
          movement,
          stress,
        };

        currentBPM.current = heartRate;
        edaLevel.current = eda;

        setHistory((old) => [
          ...old.slice(-49),
          {
            time: new Date().toLocaleTimeString(),
            ...updated,
          },
        ]);

        return updated;
      });

    }, 1000);

    return () => clearInterval(interval);

  }, []);

  // ------------------------------------------
  // 2. GENERATE CONTINUOUS LIVE SIGNALS
  // ------------------------------------------

  useEffect(() => {

    const interval = setInterval(() => {

      sampleNumber.current += 1;

      const time = sampleNumber.current;

      // PPG pulse waveform
      const bpm = currentBPM.current;

      phase.current += (2 * Math.PI * bpm) / (60 * 20);

      const ppgValue =
        0.45 * Math.sin(phase.current) +
        0.12 * Math.sin(2 * phase.current) +
        0.04 * Math.sin(3 * phase.current) +
        random(-0.025, 0.025);

      // EDA gradual fluctuations
      edaLevel.current = clamp(
        edaLevel.current + random(-0.025, 0.025),
        0.1,
        5
      );

      const edaValue =
        edaLevel.current + random(-0.03, 0.03);

      // Accelerometer gradual movement
      accel.current = {
        x: clamp(accel.current.x + random(-0.04, 0.04), -1.5, 1.5),
        y: clamp(accel.current.y + random(-0.04, 0.04), -1.5, 1.5),
        z: clamp(accel.current.z + random(-0.025, 0.025), 0.5, 1.5),
      };

      const newPPG = {
        time,
        value: Number(ppgValue.toFixed(3)),
      };

      const newEDA = {
        time,
        value: Number(edaValue.toFixed(3)),
      };

      const newAccel = {
        time,
        x: Number(accel.current.x.toFixed(3)),
        y: Number(accel.current.y.toFixed(3)),
        z: Number(accel.current.z.toFixed(3)),
      };

      setSignals((previous) => ({
        ppg: [...previous.ppg, newPPG].slice(-100),
        eda: [...previous.eda, newEDA].slice(-100),
        accelerometer: [
          ...previous.accelerometer,
          newAccel,
        ].slice(-100),
      }));

    }, 50);

    return () => clearInterval(interval);

  }, []);

  return {
    readings,
    history,
    signals,
  };
}
