import TextToSignPlayer from "./TextToSignPlayer";
import { useState, useCallback } from "react";
import debounce from "lodash/debounce";

function TextToSign() {
  const [text, setText] = useState<string[]>([]);

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
          className="textarea aspect-video h-full w-full rounded-lg border border-slate-300 bg-slate-100 shadow-sm focus:border-slate-400 focus:ring focus:ring-slate-200 focus:ring-opacity-50"
          placeholder="Enter text to translate"
          onChange={debouncedHandleSearch}
          style={{
            resize: "none",
          }}
        />
      </div>
      <div className="flex justify-center">
        <TextToSignPlayer text={text} />
      </div>
    </div>
  );
}
export default TextToSign;
