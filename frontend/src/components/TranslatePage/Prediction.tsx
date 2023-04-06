import React, { useState } from "react";
import Button from "@mui/material/Button";
import "./Prediction.css";
type PredictionProps = {
  startRef: React.MutableRefObject<boolean>;
  prediction: string[];
  loading: boolean;
};
function Prediction({ startRef, prediction, loading }: PredictionProps) {
  const [startRecording, setStartRecording] = useState<boolean>(false);
  return (
    <div
      style={{ backgroundColor: "#FFFFFF" }}
      className="flex h-full flex-col p-2 text-center "
    >
      <div className="translation-box flex flex-grow flex-col items-center justify-between rounded-xl text-white">
        <label
          style={{
            fontSize: 24,
          }}
        >
          {prediction.filter((res) => res !== "africa").join(" ")}
        </label>
        <Button
          variant="contained"
          onClick={() => {
            setStartRecording((prev) => {
              startRef.current = !prev;
              return !prev;
            });
          }}
          color="warning"
        >
          {loading ? "Loading" : startRecording ? "Stop" : "Start"}
        </Button>
      </div>
    </div>
  );
}
export default Prediction;
