import React, { useState } from "react";
import Camera from "./TranslatePage/Camera";
import Prediction from "./TranslatePage/Prediction";

function SigntoSpeech() {
  const [prediction, setPrediction] = useState<Array<string>>([]);
  const [loading, setLoading] = useState(false);
  const startRef = React.useRef(false);

  return (
    <div className="grid h-full grid-cols-1 grid-rows-2 md:grid-cols-2 md:grid-rows-1">
      <div>
        <Camera
          startRef={startRef}
          setPrediction={setPrediction}
          setLoading={setLoading}
        />
      </div>
      <div className="">
        <Prediction
          startRef={startRef}
          prediction={prediction}
          loading={loading}
        />
      </div>
    </div>
  );
}

export default SigntoSpeech;
