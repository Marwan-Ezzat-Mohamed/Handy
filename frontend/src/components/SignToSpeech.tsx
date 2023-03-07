import React, { useState } from "react";
import Camera from "./TranslatePage/Camera";
import Prediction from "./TranslatePage/Prediction";
import VideoInput from "./VideoInput";

function SigntoSpeech() {
  const [prediction, setPrediction] = useState<Array<string>>([]);
  const startRef = React.useRef(false);

  return (
    <div className="mt-2 flex h-full w-full flex-col space-y-10">
      <div className="flex w-full flex-grow space-y-10 bg-white sm:flex-col md:flex-row">
        <Camera startRef={startRef} setPrediction={setPrediction} />
        <VideoInput width={400} height={300} />
        <Prediction startRef={startRef} prediction={prediction} />
      </div>
    </div>
  );
}
export default SigntoSpeech;
