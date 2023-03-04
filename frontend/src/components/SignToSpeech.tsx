import React, { useState } from "react";
import Camera from "./TranslatePage/Camera";
import FormatBar from "./TranslatePage/FormatBar";
import Prediction from "./TranslatePage/Prediction";

function SigntoSpeech() {
  const [prediction, setPrediction] = useState<Array<string>>([]);
  const startRef = React.useRef(false);

  return (
    <div className="mt-2 flex h-full w-full flex-col space-y-10">
      <div className="flex w-full flex-grow flex-col space-y-10 bg-white">
        <Camera startRef={startRef} setPrediction={setPrediction} />
        <Prediction startRef={startRef} prediction={prediction} />
      </div>
    </div>
  );
}
export default SigntoSpeech;
