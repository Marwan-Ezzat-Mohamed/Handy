import { draw, getActionFrames } from "../utils";
import { useRef, useEffect, useCallback } from "react";

interface TextToSignPlayerProps {
  text: string[];
}

const FPS = 30;

const TextToSignPlayer = ({ text }: TextToSignPlayerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playbackSpeedRef = useRef<number>(1);
  const stopCurrentPlayingRef = useRef<boolean>(false);

  const playFrames = useCallback(async (frames: any[]) => {
    if (stopCurrentPlayingRef.current) {
      stopCurrentPlayingRef.current = false;
      return;
    }
    const canvas = canvasRef.current;

    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    //draw a frame with the playback speed with default of 1 and 30 fps
    for (const frame of frames) {
      draw(ctx, frame, false);
      await new Promise((resolve) =>
        setTimeout(resolve, 1000 / FPS / playbackSpeedRef.current)
      );
    }
    playFrames(frames);
  }, []);

  useEffect(() => {
    async function getFrames() {
      if (!text) return;
      const promises = text.map((text) => getActionFrames(text));
      const frames = await Promise.all(promises);
      //merge frames
      const mergedFrames = frames.reduce((acc, cur) => {
        return acc.concat(cur);
      }, []);

      playFrames(mergedFrames);
    }
    getFrames();

    return () => {
      stopCurrentPlayingRef.current = true;
    };
  }, [playFrames, text]);

  return (
    <div>
      <h1>{text}</h1>
      <canvas
        ref={canvasRef}
        style={{
          width: "400px",
          height: "400px",
          backgroundColor: "black",
        }}
      ></canvas>
    </div>
  );
};

export default TextToSignPlayer;
