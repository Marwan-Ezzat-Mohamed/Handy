import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import SigntoSpeech from "./components/SignToSpeech";
import SpeechtoText from "./components/SpeechToText";
import FormatBar from "./components/TranslatePage/FormatBar";

class App extends React.Component {
  render() {
    return (
      <div className="flex h-screen w-screen flex-col  bg-white">
        <BrowserRouter>
          <FormatBar />
          <Routes>
            <Route path="/" element={<SigntoSpeech />}></Route>
            <Route path="/texttosign" element={<SpeechtoText />}></Route>
          </Routes>
        </BrowserRouter>
      </div>
    );
  }
}

export default App;
