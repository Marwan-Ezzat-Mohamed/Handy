import TextToSignPlayer from "./TextToSignPlayer";
import { useState, useCallback } from "react";
import debounce from "lodash/debounce";
function TextToSign() {
  const [text, setText] = useState<string[]>([]);

  const filterSearchText = (text: string) => {
    return text
      .replace(/[^a-zA-Z0-9 ]/g, "")
      .replace(/\s+/g, " ")
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
    <div className="mt-2 flex h-full w-full  items-center space-x-10">
      <textarea
        className=" textarea aspect-video h-1/2 w-1/2 rounded-lg border border-slate-300 bg-slate-100 shadow-sm focus:border-slate-400 focus:ring focus:ring-slate-200 focus:ring-opacity-50"
        placeholder="Enter text to translate"
        onChange={debouncedHandleSearch}
      />
      <div className="h-1/2 w-1/2">
        <TextToSignPlayer text={text} />
      </div>
    </div>
  );
}
export default TextToSign;
