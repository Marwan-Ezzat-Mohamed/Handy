import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import SignToSpeech from "./components/SignToSpeech";
import TextToSign from "./components/TextToSign";
import FormatBar from "./components/TranslatePage/FormatBar";
import LessonsList from "./components/LearnPage/LessonsList";
import LessonContent from "./components/LearnPage/LessonContent";
import LessonDetails from "./components/LearnPage/LessonDetails";
import LearnPage from "./components/LearnPage/LearnPage";
import QuizContent from "./components/LearnPage/QuizContent";

class App extends React.Component {
  render() {
    return (
      <div className="main-background flex h-screen w-screen flex-col bg-white">
        <BrowserRouter>
          {/* <FormatBar /> */}
          <Routes>
            <Route path="/" element={<LearnPage />}></Route>
            <Route path="/lesson/:lesson" element={<LessonContent />}></Route>
            <Route path="/quiz" element={<QuizContent />}></Route>
            <Route path="/signtotext" element={<SignToSpeech />}></Route>
            <Route path="/texttosign" element={<TextToSign />}></Route>
          </Routes>
        </BrowserRouter>
      </div>
    );
  }
}

export default App;
