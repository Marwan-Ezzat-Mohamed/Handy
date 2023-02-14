import * as Holistic from "@mediapipe/holistic";
import * as Camera from "@mediapipe/camera_utils";
import * as drawingUtils from "@mediapipe/drawing_utils";

import { useRef, useEffect } from "react";

function MediapipeCamera({
  onResult,
}: {
  onResult: (results: Holistic.Results) => void;
}) {
  //const controls = window;
  const canvasRef = useRef<null | HTMLCanvasElement>(null);
  const videoRef = useRef(null);
  const config = {
    locateFile: (file: any) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    },
  };

  const connect = (
    ctx: CanvasRenderingContext2D,
    connectors: Array<
      [Holistic.NormalizedLandmark, Holistic.NormalizedLandmark]
    >
  ): void => {
    const canvas = ctx.canvas;
    for (const connector of connectors) {
      const from = connector[0];
      const to = connector[1];
      if (from && to) {
        if (
          from.visibility &&
          to.visibility &&
          (from.visibility < 0.1 || to.visibility < 0.1)
        ) {
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
        ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
        ctx.stroke();
      }
    }
  };

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
    const camera = new Camera.Camera(videoRef.current!, {
      onFrame: async () => {
        await holistic.send({ image: videoRef.current! });
      },
    });

    const holistic = new Holistic.Holistic(config);
    holistic.setOptions({
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    holistic.onResults(onMediapipeResults);
    camera.start();
  }, []);

  return (
    <div className="App">
      <canvas ref={canvasRef} />
      <video style={{ display: "none" }} ref={videoRef} />
    </div>
  );
}

export { MediapipeCamera };
