import { Results, NormalizedLandmark } from "@mediapipe/holistic";
import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { modelAtom } from "../../stores/jotai";
import { useAtom } from "jotai";
import { labelMap } from "../../utils/index";
import { MediapipeCamera } from "../MediapipeCamera";
import { FRAMES_FOR_PREDICTION } from "./../../utils/index";

function chunkArray(arr: any[], chunkSize: number): any[][] {
  const result = [];
  for (let i = 0; i < arr.length; i += chunkSize) {
    result.push(arr.slice(i, i + chunkSize));
  }
  return result;
}

function extractKeypoints(results: Results): number[] {
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
type CameraProps = {
  startRef: React.MutableRefObject<boolean>;
  setPrediction: React.Dispatch<React.SetStateAction<string[]>>;
};

function Camera({ startRef, setPrediction }: CameraProps) {
  const [model, setModel] = useAtom(modelAtom);
  const [results, setResults] = useState<null | Results[]>([]);

  const [predict, setPredict] = useState(false);

  useEffect(() => {
    startPrediction();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predict]);

  useEffect(() => {
    async function load() {
      const model = await tf.loadLayersModel("./model.json");
      setModel(model);
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  async function startPrediction() {
    if (results && results.length >= FRAMES_FOR_PREDICTION) {
      const filteredResults = results.filter((result) => {
        const extractedKeypoints = extractKeypoints(result);
        //check if the sum of the keypoints is 0
        const sum = extractedKeypoints.reduce((a, b) => a + b, 0);
        return sum === 0 ? false : true;
      });
      const chunks = chunkArray(filteredResults, FRAMES_FOR_PREDICTION);

      const sequences = chunks
        .map((chunk) => {
          return chunk.map((result) => extractKeypoints(result));
        })
        .filter((seq) => seq.length === FRAMES_FOR_PREDICTION);

      console.log(chunks);

      //use the layers model to predict the gesture and get the confidence score and the label
      if (model) {
        const prediction = model.predict(tf.tensor3d(sequences)) as tf.Tensor;
        prediction.data().then((data) => {
          const labels = prediction.argMax(-1).dataSync();
          //get the confidence score and the label of each gesture

          for (let i = 0; i < labels.length; i++) {
            console.log({
              word: labelMap[labels[i].toString() as keyof typeof labelMap],
              confidence: data[i],
              labels,
            });
          }

          console.log("\n\n\n");

          setPrediction((prev) => {
            const newSet = [...prev];
            labels.forEach((label: number) => {
              const word = labelMap[label.toString() as keyof typeof labelMap];
              if (newSet[newSet.length - 1] !== word) {
                newSet.push(word);
              }
            });

            return newSet;
          });
        });
      }
      setResults([]);
    }
  }
  const onResults = (results: Results) => {
    if (startRef.current) {
      //   console.log('haga');
      setResults((prev) => {
        if (prev) {
          return [...prev, results];
        }
        return [results];
      });
      setPrediction([]);
    } else {
      setPredict((prev) => !prev);
    }
  };

  return (
    <div className="flex w-full flex-grow flex-col  bg-white p-2 text-center">
      <MediapipeCamera onResult={onResults} />
    </div>
  );
}
export default Camera;
