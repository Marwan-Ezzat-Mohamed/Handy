import { NormalizedLandmark } from "@mediapipe/holistic";
import { FRAMES_FOR_PREDICTION, labelMap } from "../../utils/index";
import * as tf from "@tensorflow/tfjs";
import { PredictType } from "../../types";

function chunkArray(arr: any[], chunkSize: number): any[][] {
  const result = [];
  for (let i = 0; i < arr.length; i += chunkSize) {
    result.push(arr.slice(i, i + chunkSize));
  }
  return result;
}

function extractKeypoints(results: PredictType): number[] {
  let lh: number[] = [];
  if (results.leftHandLandmarks) {
    lh = lh.concat(
      ...results.leftHandLandmarks.map((res: NormalizedLandmark) => [
        res.x,
        res.y,
        res.z,
      ])
    );
  } else {
    lh = Array(21 * 3).fill(0);
  }

  let rh: number[] = [];
  if (results.rightHandLandmarks) {
    rh = rh.concat(
      ...results.rightHandLandmarks.map((res: NormalizedLandmark) => [
        res.x,
        res.y,
        res.z,
      ])
    );
  } else {
    rh = Array(21 * 3).fill(0);
  }
  const final = lh.concat(rh);
  return final;
}

async function startPrediction(results: PredictType[]): Promise<string[]> {
  tf.setBackend("cpu");
  const model = await tf.loadLayersModel("/model.json");
  if (results && results.length >= FRAMES_FOR_PREDICTION) {
    const filteredResults = results.filter((result) => {
      const extractedKeypoints = extractKeypoints(result);
      const sum = extractedKeypoints.reduce((a, b) => a + b, 0);
      return sum === 0 ? false : true;
    });
    const chunks = chunkArray(filteredResults, FRAMES_FOR_PREDICTION);

    const sequences = chunks
      .map((chunk) => {
        return chunk.map((result) => extractKeypoints(result));
      })
      .filter((seq) => seq.length === FRAMES_FOR_PREDICTION);

    //use the layers model to predict the gesture and get the confidence score and the label
    if (model) {
      const prediction = model!.predict(tf.tensor3d(sequences)) as tf.Tensor;
      const labels = Array.from(prediction.argMax(-1).dataSync());
      const res: string[] = [];
      for (const label of labels) {
        const word = labelMap[label.toString() as keyof typeof labelMap];
        if (res[res.length - 1] !== word) {
          res.push(word);
        }
      }
      return res;
    }
    return [];
  }
  return [];
}

/*eslint-disable no-restricted-globals */
self.onmessage = function (event) {
  const data = event.data;
  const { results } = data;
  startPrediction(results).then((res) => {
    self.postMessage(res);
  });
};
