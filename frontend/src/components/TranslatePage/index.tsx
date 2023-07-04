import { useState } from "react";
import SigntoSpeech from "../SignToSpeech";
import FormatBar from "./FormatBar";
import TextToSign from "../TextToSign";

const TranslatePage = () => {
  const [translationDirection, setTranslationDirection] = useState<
    "signToText" | "textToSign"
  >("signToText");
  return (
    <div className="flex flex-grow flex-col">
      <FormatBar
        setTranslationDirection={setTranslationDirection}
        translationDirection={translationDirection}
      />
      {translationDirection === "signToText" ? (
        <SigntoSpeech />
      ) : (
        <TextToSign />
      )}
    </div>
  );
};

export default TranslatePage;
