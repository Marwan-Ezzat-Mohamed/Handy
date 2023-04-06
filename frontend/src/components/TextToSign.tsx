import TextToSignPlayer from "./TextToSignPlayer";
import { useState, useCallback } from "react";
import debounce from "lodash/debounce";

function TextToSign() {
  const [text, setText] = useState<string[]>([]);

  const filterSearchText = (text: string) => {
    return text
      .replace("the", "")
      .replace("is", " ")
      .replace("dancing", "dance")
      .trim();
  };
  const handleSearch = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setText(e.target.value.split(" "));
    },
    []
  );
  const debouncedHandleSearch = useCallback(debounce(handleSearch, 200), []);

  return (
    <div className="grid  h-1/2 grid-cols-1  justify-center gap-4 px-8 md:grid-cols-2">
      <div className="flex justify-center">
        <textarea
          style={{
            color: "#000000",
            backgroundColor: "#fff6df",
            resize: "none",
            maxHeight: "400px",
          }}
          className="textarea aspect-video h-full w-full rounded-lg"
          placeholder="Enter text to translate"
          onChange={debouncedHandleSearch}
        />
      </div>
      <div className="flex justify-center">
        <TextToSignPlayer text={text} />
      </div>
    </div>
  );
}
export default TextToSign;
