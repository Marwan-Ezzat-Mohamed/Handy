import React, { useState } from "react";
import Button from "@mui/material/Button";
type PredictionProps = {
  startRef: React.MutableRefObject<boolean>;
  prediction: string[];
};
function Prediction({ startRef, prediction }: PredictionProps) {
  const [startRecording, setStartRecording] = useState<boolean>(false);
  return (
    <div className="flex h-full flex-col bg-white p-2 text-center ">
      <div className="flex flex-grow flex-col items-center justify-between rounded-xl bg-indigo-600 text-white">
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
          {startRecording ? "Stop" : "Start"}
        </Button>
      </div>
    </div>
  );
}
export default Prediction;
