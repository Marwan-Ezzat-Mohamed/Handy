import TextToSignPlayer from "./TextToSignPlayer";
import { useState, useCallback } from "react";
import debounce from "lodash/debounce";

function TextToSign() {
  const [text, setText] = useState<string[]>([]);

  const filterSearchText = (text: string[]) => {
    return text
      .join(" ")
      .replace("the", "")
      .replace("is", " ")
      .replace("dancing", "dance")
      .split(" ")
      .filter((word) => word !== "");
  };
  const handleSearch = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setText(e.target.value.split(" "));
    },
    []
  );
  const debouncedHandleSearch = useCallback(debounce(handleSearch, 200), []);

  return (
    <div className="grid  h-1/2 grid-cols-1 justify-center gap-4 px-8 md:grid-cols-2">
      <div className="flex justify-center">
        <textarea
          style={{
            resize: "none",
            height: "clamp(150px, 90%, 520px)",
          }}
          className="textarea aspect-video h-full w-full rounded-lg bg-yellow-50 text-2xl font-extrabold text-black"
          placeholder="Enter text to translate"
          onChange={debouncedHandleSearch}
        />
      </div>
      <div className="flex justify-center">
        <TextToSignPlayer text={filterSearchText(text)} />
      </div>
    </div>
  );
}
export default TextToSign;
