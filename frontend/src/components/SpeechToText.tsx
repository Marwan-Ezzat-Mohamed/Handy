import React, { useState } from "react";
import FormatBar from "./TranslatePage/FormatBar";
import SearchBox from "./SearchBox";
function SpeechtoText() {
  return (
    <div className="mt-2 flex h-full w-full flex-col space-y-10">
      <SearchBox />
    </div>
  );
}
export default SpeechtoText;
