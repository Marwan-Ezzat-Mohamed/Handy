import React, { useState } from "react";
import Button from "@mui/material/Button";
type PredictionProps = {
  startRef: React.MutableRefObject<boolean>;
  prediction: string[];
};
function Prediction({ startRef, prediction }: PredictionProps) {
  const [startRecording, setStartRecording] = useState<boolean>(false);
  return (
    <div className="inline-block  h-2/5 w-full  space-y-4 bg-white text-center">
      <div className="inline-block h-4/5 w-4/5 rounded-xl bg-indigo-600  text-white ">
        <label
          style={{
            fontSize: 24,
          }}
        >
          {prediction.filter((res) => res !== "africa").join(" ")}
          word
        </label>
      </div>
      <div className=" h-1/5 w-full">
        <Button
          variant="contained"
          onClick={() => {
            setStartRecording((prev) => {
              startRef.current = !prev;
              return !prev;
            });
          }}
        >
          {startRecording ? "Stop" : "Start"}
        </Button>
      </div>
    </div>
  );
}
export default Prediction;
