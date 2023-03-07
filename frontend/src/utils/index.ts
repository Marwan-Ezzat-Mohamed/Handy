// you need to install tensorflowjs
// pip install tensorflowjs
//this is use to convert keras model.h5 to tensorflowjs model
// tensorflowjs_converter --input_format=keras model.h5 .

import * as Holistic from "@mediapipe/holistic";

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
