import * as Holistic from "@mediapipe/holistic";
import * as drawingUtils from "@mediapipe/drawing_utils";
import { useMemo } from "react";

import { useRef, useEffect, useState } from "react";
import { connect } from "../../utils";
let holistic: Holistic.Holistic | null = null;

function MediapipeCamera({
  onResult,
}: {
  onResult: (results: Holistic.Results) => void;
}) {
  //const controls = window;

  const [uploadedVideo, setUploadedVideo] = useState<any>(null);
  const [useCamera, setUseCamera] = useState<boolean>(true);
  const canvasRef = useRef<null | HTMLCanvasElement>(null);
  const videoRef = useRef<any>(null);
  const config = useMemo(
    () => ({
      locateFile: (file: any) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
      },
    }),
    []
  );

  let activeEffect = "mask";

  const onMediapipeResults = (results: Holistic.Results): void => {
    onResult(results);
    if (!canvasRef.current) return;
    const canvasCtx = canvasRef.current.getContext("2d");
    if (!canvasCtx) return;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasCtx.canvas.width, canvasCtx.canvas.height);
    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasCtx.canvas.width,
      canvasCtx.canvas.height
    );
    if (results.segmentationMask) {
      canvasCtx.drawImage(
        results.segmentationMask,
        0,
        0,
        canvasCtx.canvas.width,
        canvasCtx.canvas.height
      );
      if (activeEffect === "mask" || activeEffect === "both") {
        canvasCtx.globalCompositeOperation = "source-in";
        canvasCtx.fillStyle = "#00FF007F";
        canvasCtx.fillRect(
          0,
          0,
          canvasCtx.canvas.width,
          canvasCtx.canvas.height
        );
      } else {
        canvasCtx.globalCompositeOperation = "source-out";
        canvasCtx.fillStyle = "#0000FF7F";
        canvasCtx.fillRect(
          0,
          0,
          canvasCtx.canvas.width,
          canvasCtx.canvas.height
        );
      }
      canvasCtx.globalCompositeOperation = "destination-atop";

      canvasCtx.globalCompositeOperation = "source-over";
    } else {
    }
    canvasCtx.lineWidth = 5;
    if (results.poseLandmarks) {
      if (results.rightHandLandmarks) {
        canvasCtx.strokeStyle = "black";
        connect(canvasCtx, [
          [
            results.poseLandmarks[Holistic.POSE_LANDMARKS.RIGHT_ELBOW],
            results.rightHandLandmarks[0],
          ],
        ]);
      }
      if (results.leftHandLandmarks) {
        canvasCtx.strokeStyle = "black";
        connect(canvasCtx, [
          [
            results.poseLandmarks[Holistic.POSE_LANDMARKS.LEFT_ELBOW],
            results.leftHandLandmarks[0],
          ],
        ]);
      }
      drawingUtils.drawConnectors(
        canvasCtx,
        results.poseLandmarks,
        Holistic.POSE_CONNECTIONS,
        { color: "black" }
      );
      drawingUtils.drawLandmarks(
        canvasCtx,
        Object.values(Holistic.POSE_LANDMARKS_LEFT).map(
          (index) => results.poseLandmarks[index]
        ),
        { visibilityMin: 0.65, color: "black", fillColor: "rgb(255,138,0)" }
      );
      drawingUtils.drawLandmarks(
        canvasCtx,
        Object.values(Holistic.POSE_LANDMARKS_RIGHT).map(
          (index) => results.poseLandmarks[index]
        ),
        { visibilityMin: 0.65, color: "black", fillColor: "rgb(0,217,231)" }
      );

      // Hands...
      drawingUtils.drawConnectors(
        canvasCtx,
        results.rightHandLandmarks,
        Holistic.HAND_CONNECTIONS,
        { color: "black" }
      );
      drawingUtils.drawLandmarks(canvasCtx, results.rightHandLandmarks, {
        color: "black",
        fillColor: "rgb(0,217,231)",
        lineWidth: 2,
        radius: (data: any) => {
          return drawingUtils.lerp(data.from!.z!, -0.15, 0.1, 10, 1);
        },
      });
      drawingUtils.drawConnectors(
        canvasCtx,
        results.leftHandLandmarks,
        Holistic.HAND_CONNECTIONS,
        { color: "black" }
      );
      drawingUtils.drawLandmarks(canvasCtx, results.leftHandLandmarks, {
        color: "black",
        fillColor: "rgb(255,138,0)",
        lineWidth: 2,
        radius: (data: any) => {
          return drawingUtils.lerp(data.from!.z!, -0.15, 0.1, 10, 1);
        },
      });

      // Face...
      drawingUtils.drawConnectors(
        canvasCtx,
        results.faceLandmarks,
        Holistic.FACEMESH_TESSELATION,
        { color: "#C0C0C070", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        canvasCtx,
        results.faceLandmarks,
        Holistic.FACEMESH_RIGHT_EYE,
        { color: "rgb(0,217,231)" }
      );
      drawingUtils.drawConnectors(
        canvasCtx,
        results.faceLandmarks,
        Holistic.FACEMESH_RIGHT_EYEBROW,
        { color: "rgb(0,217,231)" }
      );
      drawingUtils.drawConnectors(
        canvasCtx,
        results.faceLandmarks,
        Holistic.FACEMESH_LEFT_EYE,
        { color: "rgb(255,138,0)" }
      );
      drawingUtils.drawConnectors(
        canvasCtx,
        results.faceLandmarks,
        Holistic.FACEMESH_LEFT_EYEBROW,
        { color: "rgb(255,138,0)" }
      );
      drawingUtils.drawConnectors(
        canvasCtx,
        results.faceLandmarks,
        Holistic.FACEMESH_FACE_OVAL,
        { color: "#E0E0E0", lineWidth: 5 }
      );
      drawingUtils.drawConnectors(
        canvasCtx,
        results.faceLandmarks,
        Holistic.FACEMESH_LIPS,
        { color: "#E0E0E0", lineWidth: 5 }
      );

      canvasCtx.restore();
    }
  };
  useEffect(() => {
    holistic = new Holistic.Holistic(config);
    holistic.setOptions({
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    holistic.onResults(onMediapipeResults);
  }, []);

  useEffect(() => {
    const video = videoRef.current!;
    const onFrame = () => {
      const renderCanvas = async () => {
        if (!videoRef.current || !holistic) return;
        if (video.paused || video.ended) {
          return;
        }
        await holistic.send({ image: video });
        requestAnimationFrame(renderCanvas);
      };
      renderCanvas();
    };

    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: 640,
          height: 480,
        },
      })
      .then((stream) => {
        if (videoRef.current && useCamera) {
          videoRef.current.srcObject = stream;
        } else {
          videoRef.current.srcObject = null;
          if (canvasRef.current) {
            const canvasCtx = canvasRef.current.getContext("2d");
            if (!canvasCtx) return;
            canvasCtx.save();
            canvasCtx.clearRect(
              0,
              0,
              canvasCtx.canvas.width,
              canvasCtx.canvas.height
            );
          }

          videoRef.current.src = uploadedVideo;
        }
      });

    video.addEventListener("play", onFrame);

    return () => {
      video.removeEventListener("play", onFrame);
    };
  }, [useCamera, uploadedVideo]);

  return (
    <div className="flex h-full flex-col items-center justify-center rounded-xl bg-black">
      <canvas
        className="flex aspect-video rounded-xl"
        ref={canvasRef}
        style={{
          maxHeight: "300px",
          maxWidth: "100%",
          minWidth: "100%",
        }}
      />

      <div>
        <button
          className="bg-slate-900 text-white"
          onClick={() => {
            setUseCamera((prev) => !prev);
          }}
        >
          {!useCamera ? "Using Video" : "Using Camera"}
        </button>

        <input
          type="file"
          accept={useCamera ? "video/*;capture=camera" : "video/*"}
          onChange={(e) => {
            const file = e.target!.files![0];
            const url = URL.createObjectURL(file);
            setUploadedVideo(url);
          }}
        />
      </div>
      <video
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "0px ",
          height: "0px",
        }}
        ref={videoRef}
        src={uploadedVideo}
        muted
        loop
        autoPlay={true}
      />
    </div>
  );
}

export { MediapipeCamera };
