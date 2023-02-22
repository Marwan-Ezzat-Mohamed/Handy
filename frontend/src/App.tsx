import React, { useEffect, useState } from "react";
import SigntoSpeech from "./components/SignToSpeech";

function App() {
  return (
    <div className="flex h-screen w-screen flex-col  bg-white">
      <SigntoSpeech />
    </div>
  );
}

export default App;
