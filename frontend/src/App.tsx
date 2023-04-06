import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import SignToSpeech from "./components/SignToSpeech";
import TextToSign from "./components/TextToSign";
import FormatBar from "./components/TranslatePage/FormatBar";

class App extends React.Component {
  render() {
    return (
      <div className="main-background flex h-screen w-screen flex-col bg-white">
        <BrowserRouter>
          <FormatBar />
          <Routes>
            <Route path="/" element={<SignToSpeech />}></Route>
            <Route path="/texttosign" element={<TextToSign />}></Route>
          </Routes>
        </BrowserRouter>
      </div>
    );
  }
}

export default App;
