// you need to install tensorflowjs
// pip install tensorflowjs
//this is use to convert keras model.h5 to tensorflowjs model
// tensorflowjs_converter --input_format=keras model.h5 .

import * as Holistic from "@mediapipe/holistic";
import * as drawing from "@mediapipe/drawing_utils";

export interface Pose {
  faceLandmarks: PoseLandmark[];
  poseLandmarks: PoseLandmark[];
  rightHandLandmarks: PoseLandmark[];
  leftHandLandmarks: PoseLandmark[];
  image: HTMLCanvasElement;
}

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export const labelMap = {
  "0": "accident",
  "1": "adopt",
  "2": "afraid",
  "3": "africa",
  "4": "again",
  "5": "alarm",
  "6": "all",
  "7": "apple",
  "8": "awful",
  "9": "bad",
  "10": "banana",
  "11": "bathroom",
  "12": "beautiful",
  "13": "before",
  "14": "bicycle",
  "15": "big",
  "16": "bird",
  "17": "black",
  "18": "blue",
  "19": "book",
  "20": "bored",
  "21": "boss",
  "22": "boy",
  "23": "bread",
  "24": "brother",
  "25": "brown",
  "26": "build",
  "27": "but",
  "28": "buy",
  "29": "can",
  "30": "candy",
  "31": "car",
  "32": "cat",
  "33": "cheese",
  "34": "church",
  "35": "city",
  "36": "class",
  "37": "clothes",
  "38": "coffee",
  "39": "cold",
};

export const connect = (
  ctx: CanvasRenderingContext2D,
  connectors: Array<[Holistic.NormalizedLandmark, Holistic.NormalizedLandmark]>
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
const IGNORED_BODY_LANDMARKS = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22,
];

function drawBody(
  landmarks: PoseLandmark[],
  ctx: CanvasRenderingContext2D
): void {
  const filteredLandmarks = Array.from(landmarks);
  for (const l of IGNORED_BODY_LANDMARKS) {
    delete filteredLandmarks[l];
  }

  drawing.drawConnectors(ctx, filteredLandmarks, Holistic.POSE_CONNECTIONS, {
    color: "#00FF00",
  });
  drawing.drawLandmarks(ctx, filteredLandmarks, {
    color: "#00FF00",
    fillColor: "#FF0000",
  });
}
function drawNeck(
  landmarks: PoseLandmark[],
  ctx: CanvasRenderingContext2D,
  lineColor: string
) {
  // get the left and right shoulder landmarks
  const leftShoulder = landmarks[Holistic.POSE_LANDMARKS.LEFT_SHOULDER];
  const rightShoulder = landmarks[Holistic.POSE_LANDMARKS.RIGHT_SHOULDER];
  const head = landmarks[Holistic.POSE_LANDMARKS.NOSE];
  // draw a vertical line between the two shoulders to the head
  ctx.beginPath();
  const x = (leftShoulder.x + rightShoulder.x) / 2;
  const y = (leftShoulder.y + rightShoulder.y) / 2;
  ctx.moveTo(x * ctx.canvas.width, y * ctx.canvas.height);
  ctx.lineTo(head.x * ctx.canvas.width, head.y * ctx.canvas.height);
  ctx.strokeStyle = lineColor;
  ctx.stroke();
}

function drawHand(
  landmarks: PoseLandmark[],
  ctx: CanvasRenderingContext2D,
  lineColor: string,
  dotColor: string,
  dotFillColor: string
): void {
  const fingerColors = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF"];
  drawConnect([[landmarks[0], landmarks[9]]], ctx, fingerColors[2]); //middle finger
  drawConnect([[landmarks[0], landmarks[13]]], ctx, fingerColors[3]); //ring finger

  drawing.drawConnectors(ctx, landmarks, Holistic.HAND_CONNECTIONS, {
    color: (info) => {
      const ignore = [8, 12, 16];
      if (ignore.includes(info.index ?? 0)) return "transparent";
      return fingerColors[Math.floor((info.index ?? 0) / 4)];
    },
  });

  //   for (let i = 0; i < 5; i++) {
  //     const fingerLandmarks = landmarks.slice(i * 4 + 1, i * 4 + 5); // get landmarks for current finger
  //     drawing.drawLandmarks(ctx, fingerLandmarks, {
  //       color: fingerColors[i], // set color to color of current finger
  //       fillColor: dotFillColor,
  //       lineWidth: 2,
  //       radius: (landmark) => {
  //         return drawing.lerp(1, -0.15, 0.01, 10, 1);
  //       },
  //     });
  //   }
}

function drawFace(
  landmarks: PoseLandmark[],
  ctx: CanvasRenderingContext2D
): void {
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_TESSELATION, {
    color: "#C0C0C070",
    lineWidth: 1,
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_RIGHT_EYE, {
    color: "#FF3030",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_RIGHT_EYEBROW, {
    color: "#FF3030",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_LEFT_EYE, {
    color: "#30FF30",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_LEFT_EYEBROW, {
    color: "#30FF30",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_FACE_OVAL, {
    color: "#E0E0E0",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_LIPS, {
    color: "#E0E0E0",
  });
}

function drawConnect(
  connectors: PoseLandmark[][],
  ctx: CanvasRenderingContext2D,
  color = "#00FF00"
): void {
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
      ctx.moveTo(from.x * ctx.canvas.width, from.y * ctx.canvas.height);
      ctx.lineTo(to.x * ctx.canvas.width, to.y * ctx.canvas.height);
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }
}

function drawElbowHandsConnection(
  pose: Pose,
  ctx: CanvasRenderingContext2D
): void {
  ctx.lineWidth = 5;

  if (pose.rightHandLandmarks) {
    ctx.strokeStyle = "#00FF00";
    drawConnect(
      [
        [
          pose.poseLandmarks[Holistic.POSE_LANDMARKS.RIGHT_ELBOW],
          pose.rightHandLandmarks[0],
        ],
      ],
      ctx
    );
  }

  if (pose.leftHandLandmarks) {
    ctx.strokeStyle = "#FF0000";
    drawConnect(
      [
        [
          pose.poseLandmarks[Holistic.POSE_LANDMARKS.LEFT_ELBOW],
          pose.leftHandLandmarks[0],
        ],
      ],
      ctx
    );
  }
}

export async function draw(
  ctx: CanvasRenderingContext2D,
  pose: Holistic.Results,
  drawImage = true
) {
  if (!ctx) return;
  ctx.save();
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (drawImage) {
    ctx.drawImage(pose.image, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  console.log({ pose });
  if (pose.poseLandmarks) {
    drawBody(pose.poseLandmarks, ctx);
    drawElbowHandsConnection(pose as Pose, ctx);
    drawNeck(pose.poseLandmarks, ctx, "#00FFFF");
  }

  if (pose.leftHandLandmarks) {
    drawHand(pose.leftHandLandmarks, ctx, "#CC0000", "#FF0000", "#00FF00");
  }

  if (pose.rightHandLandmarks) {
    drawHand(pose.rightHandLandmarks, ctx, "#00CC00", "#00FF00", "#FF0000");
  }

  if (pose.faceLandmarks) {
    console.log("face");
    drawFace(pose.faceLandmarks, ctx);
  }

  ctx.restore();
}

export const getActionFrames = async (text: string) => {
  //load json file from public folder in react app
  const json = await fetch("/keypoints_reshaped.json");
  const data = await json.json();
  const frames = data[text];
  return frames;
};
