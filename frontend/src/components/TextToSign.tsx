
import SearchBox from "./SearchBox";
import TextToSignPlayer from "./TextToSignPlayer";
function TextToSign() {
  return (
    <div className="mt-2 flex h-full w-full flex-col space-y-10">
      <SearchBox />
      <TextToSignPlayer text={["#asl", "#dog"]} />
    </div>
  );
}
export default TextToSign;
