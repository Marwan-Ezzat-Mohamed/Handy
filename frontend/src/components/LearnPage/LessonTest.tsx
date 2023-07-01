import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import { IconButton } from "@mui/material";
import { useNavigate } from "react-router-dom";
import { MediapipeCamera } from "../MediapipeCamera";
import Camera from "../TranslatePage/Camera";
import { useEffect, useRef, useState } from "react";
function LessonTest() {
  const startRef = useRef<boolean>(false);

  const [prediction, setPrediction] = useState<Array<string>>([]);

  return (
    <div className="flex h-full w-full flex-col">
      <Camera
        startRef={startRef}
        setPrediction={setPrediction}
        setLoading={() => {}}
      />
      {prediction.length ? (
        prediction.includes("teacher") ? (
          <div className="flex justify-center">You are correct</div>
        ) : (
          <div className="flex justify-center">You are wrong</div>
        )
      ) : (
        <div className="flex justify-center">No prediction</div>
      )}
    </div>
  );
}
export default LessonTest;
