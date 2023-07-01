import TextToSignPlayer from "../TextToSignPlayer";
import ArrowCircleRightIcon from "@mui/icons-material/ArrowCircleRight";
import ArrowCircleLeftIcon from "@mui/icons-material/ArrowCircleLeft";
import SignLanguageIcon from "@mui/icons-material/SignLanguage";
import { useNavigate, useParams } from "react-router-dom";
import React, { useEffect, useState } from "react";
import Camera from "../TranslatePage/Camera";
import LessonTest from "./LessonTest";
import useGlobalStore from "../../stores/zustand";

function LessonContent() {
  //get the lesson name from the url
  const { lesson } = useParams<{ lesson: string }>();
  const lessonWords = useGlobalStore((state) => state.lessons[lesson!]);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);

  const navigate = useNavigate();
  const [isLearn, setIsLearn] = React.useState(true);
  const [prediction, setPrediction] = React.useState<Array<string>>([]);
  const [loading, setLoading] = React.useState(false);
  const startRef = React.useRef(false);

  useEffect(() => {
    setCurrentWordIndex(0);
  }, [lessonWords]);

  return (
    <div className="flex h-full flex-col bg-slate-50  px-5 py-10">
      <div className="flex items-center justify-between ">
        <button
          className=" text-4xl text-primary"
          onClick={() => navigate("/")}
        >
          <ArrowCircleLeftIcon fontSize="inherit" />
        </button>
        <h1 className="text-2xl font-bold text-primary">{lesson}</h1>
      </div>
      <div className="flex flex-grow flex-col justify-between">
        <div className="flex w-full justify-center rounded-md bg-yellow-50 p-2 text-center">
          <h1 className="text-5xl font-extrabold text-black ">
            {lessonWords?.[currentWordIndex]}
          </h1>
        </div>
        <div className=" my-5 flex  flex-grow items-center justify-center">
          {/* {isLearn && <TextToSignPlayer text={["dance"]} />} */}
          {<LessonTest />}
        </div>
        <div className="flex justify-center">
          <button
            className="btn  my-6 border-warning  bg-warning text-white"
            onClick={() => setIsLearn((prev) => !prev)}
          >
            {isLearn ? "Test" : "Learn"}
            <SignLanguageIcon />
          </button>
        </div>
        <div className="flex items-center justify-between  text-center ">
          <button
            className="btn border-primary bg-primary  text-white"
            onClick={() =>
              setCurrentWordIndex((prev) => {
                if (prev === 0) {
                  return prev;
                }
                return prev - 1;
              })
            }
          >
            <ArrowCircleLeftIcon />
            {"prev"}
          </button>
          <button
            className="btn border-primary bg-primary  text-white"
            onClick={() =>
              setCurrentWordIndex((prev) => {
                if (prev === lessonWords.length - 1) {
                  return prev;
                }
                return prev + 1;
              })
            }
          >
            {"next"}
            <ArrowCircleRightIcon />
          </button>
        </div>
      </div>
    </div>
  );
}
export default LessonContent;
