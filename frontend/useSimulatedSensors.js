
import { useState, useEffect } from "react";

const randomChange = (value, min, max, change) => {
  const updated = value + (Math.random() - 0.5) * change;

  return Math.min(max, Math.max(min, updated));
};

export default function useSimulatedSensors() {

  const [readings, setReadings] = useState({
    heartRate: 76,
    eda: 1.2,
    temperature: 33.0,
    movement: 0.25,
  });

  const [history, setHistory] = useState([]);

  useEffect(() => {

    const interval = setInterval(() => {

      setReadings((previous) => {

        const updated = {
          heartRate: Math.round(
            randomChange(previous.heartRate, 55, 130, 8)
          ),

          eda: Number(
            randomChange(previous.eda, 0.1, 5, 0.5).toFixed(2)
          ),

          temperature: Number(
            randomChange(previous.temperature, 28, 38, 0.3).toFixed(1)
          ),

          movement: Number(
            randomChange(previous.movement, 0, 2, 0.3).toFixed(2)
          ),
        };

        setHistory((oldHistory) => [
          ...oldHistory.slice(-19),
          {
            time: new Date().toLocaleTimeString(),
            ...updated,
          },
        ]);

        return updated;
      });

    }, 2000);

    return () => clearInterval(interval);

  }, []);

  return { readings, history };
}
