import TextToSignPlayer from "../TextToSignPlayer";
import ArrowCircleRightIcon from "@mui/icons-material/ArrowCircleRight";
import ArrowCircleLeftIcon from "@mui/icons-material/ArrowCircleLeft";
import SignLanguageIcon from "@mui/icons-material/SignLanguage";
import { useNavigate } from "react-router-dom";
import React from "react";
import Camera from "../TranslatePage/Camera";

function LessonContent() {
  const navigate = useNavigate();
  const [currentView, setCurrentView] = React.useState("sample");
  const [prediction, setPrediction] = React.useState<Array<string>>([]);
  const [loading, setLoading] = React.useState(false);
  const startRef = React.useRef(false);
  return (
    <div className="flex h-full flex-col bg-slate-50  ">
      <div className="flex h-1/6 items-center justify-between ">
        <button className="mx-10 text-primary" onClick={() => navigate("/")}>
          <ArrowCircleLeftIcon fontSize="large" />
        </button>
        <h1 className="mx-10 text-lg font-bold text-primary">Lesson 1</h1>
      </div>
      <div className="flex h-4/5 flex-col ">
        <div className="mx-auto  flex h-1/6 w-1/2 items-center rounded-md bg-yellow-50  text-center align-middle ">
          <h1 className=" mx-auto text-5xl font-extrabold text-black ">
            {" "}
            Word{" "}
          </h1>
        </div>
        <div className=" my-5 mx-auto h-3/5  w-2/3 ">
          {currentView == "sample" && <TextToSignPlayer text={["dance"]} />}
          {currentView == "test" && <h1>Soon</h1>}
        </div>
        <div className="mx-auto flex h-1/6 w-2/3 items-center justify-between  text-center">
          {currentView === "sample" && (
            <button
              className="btn-lg btn  border-warning bg-warning text-white"
              onClick={() => setCurrentView("test")}
            >
              {"Test"}
              <SignLanguageIcon />
            </button>
          )}
          {currentView === "test" && (
            <button
              className="btn-lg btn  border-warning bg-warning text-white"
              onClick={() => setCurrentView("sample")}
            >
              {"Learn"}
              <SignLanguageIcon />
            </button>
          )}
          <button className="btn-lg btn  border-primary bg-primary text-white">
            {"next"}
            <ArrowCircleRightIcon />
          </button>
        </div>
      </div>
    </div>
  );
}
export default LessonContent;
