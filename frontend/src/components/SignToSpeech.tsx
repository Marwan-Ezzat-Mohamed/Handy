import React, { useState } from "react";
import Camera from "./TranslatePage/Camera";
import Prediction from "./TranslatePage/Prediction";
import VideoInput from "./VideoInput";

function SigntoSpeech() {
  const [prediction, setPrediction] = useState<Array<string>>([]);
  const startRef = React.useRef(false);

  return (
    <div className=" h-full w-full flex-grow  justify-center bg-white sm:block md:flex">
      <div className="flex h-1/2 max-w-2xl flex-col bg-black md:w-1/2">
        <Camera startRef={startRef} setPrediction={setPrediction} />
      </div>
      <div className="h-1/2 max-w-2xl md:w-1/2">
        <Prediction startRef={startRef} prediction={prediction} />
      </div>
      {/* <VideoInput width={400} height={300} /> */}
    </div>
  );
}
export default SigntoSpeech;
