import React, { useEffect, useState } from "react";
import SigntoSpeech from "./components/SignToSpeech";

function App() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        width: "100vw",
        backgroundColor: "black",
        color: "white",
      }}
    >
      <SigntoSpeech />
    </div>
  );
}

export default App;
