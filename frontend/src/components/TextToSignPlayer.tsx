import { draw, drawFramesAndCreateVideo, getActionFrames } from "../utils";
import { useRef, useEffect, useCallback, useState } from "react";

interface TextToSignPlayerProps {
  text: string[];
}

const DEFAULT_PLAYBACK_SPEED = 1;

const TextToSignPlayer = ({ text }: TextToSignPlayerProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(
    DEFAULT_PLAYBACK_SPEED
  );

  const [videoUrl, setVideoUrl] = useState<string>("");

  useEffect(() => {
    async function getFrames() {
      const generatedVideoUrl = await drawFramesAndCreateVideo(
        text,
        DEFAULT_PLAYBACK_SPEED
      );
      setVideoUrl(generatedVideoUrl);
    }
    getFrames();
  }, [text]);

  useEffect(() => {
    if (!videoRef.current) return;
    videoRef.current.playbackRate = playbackSpeed;
  }, [playbackSpeed]);

  return (
    <div className="flex h-full flex-col items-center ">
      <div
        className="rounded-x flex flex-grow flex-col items-center justify-center text-white"
        style={{
          height: "clamp(150px, 90%, 95%)",
        }}
      >
        <video
          style={{
            width: "100%",
            height: "100%",
          }}
          ref={videoRef}
          src={videoUrl}
          className="aspect-video rounded-xl"
          loop
          autoPlay
        />
      </div>

      <div className="">
        <input
          type="range"
          min="0.5"
          max="2.5"
          value={playbackSpeed}
          onChange={(e) => {
            setPlaybackSpeed(parseFloat(e.target.value));
          }}
          className="range range-info"
          step="0.5"
        />
        <div className="flex  justify-between space-x-4 px-2 text-xs">
          <label className="text-xl"> 0.5x</label>
          <label className="text-xl"> 1x</label>
          <label className="text-xl"> 1.5x</label>
          <label className="text-xl"> 2x</label>
          <label className="text-xl"> 2.5x</label>
        </div>
      </div>
    </div>
  );
};

export default TextToSignPlayer;
