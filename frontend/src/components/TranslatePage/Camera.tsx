import { Results, NormalizedLandmark } from "@mediapipe/holistic";
import React, { useEffect, useRef, useState, useMemo } from "react";
import * as tf from "@tensorflow/tfjs";
import { modelAtom } from "../../stores/jotai";
import { useAtom } from "jotai";
import { labelMap } from "../../utils/index";
import { MediapipeCamera } from "../MediapipeCamera";
import { FRAMES_FOR_PREDICTION } from "../../utils/index";
import { createWorkerFactory, useWorker } from "@shopify/react-web-worker";
const createWorker = createWorkerFactory(() => import("./predict"));

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
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
};

const Camera = ({ startRef, setPrediction, setLoading }: CameraProps) => {
  const [model, setModel] = useAtom(modelAtom);
  const resultsRef = useRef<Results[]>([]);
  const [predict, setPredict] = useState(false);
  const worker = useWorker(createWorker);

  useEffect(() => {
    async function load() {
      const model = await tf.loadLayersModel("./model.json");
      setModel(model);
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  async function startPrediction(results: Results[]): Promise<any[]> {
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

      console.log(chunks);

      //use the layers model to predict the gesture and get the confidence score and the label
      if (model) {
        const prediction = model!.predict(tf.tensor3d(sequences)) as tf.Tensor;
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
          //console.log(data);

          const array: any = [];
          for (let i = 0; i < labels.length; i++) {
            const word =
              labelMap[labels[i].toString() as keyof typeof labelMap];
            if (array[array.length - 1] !== word) {
              array.push(word);
            }
          }
          /*setPrediction((prev) => {
            const newSet = [...prev];
            labels.forEach((label: number) => {
              const word = labelMap[label.toString() as keyof typeof labelMap];
              if (newSet[newSet.length - 1] !== word) {
                newSet.push(word);
              }
            });

            return newSet;
          });*/
          return array;
        });
      }
      return [];
    }
    return [];
  }
  const onResults = async (res: Results) => {
    if (startRef.current) {
      //   console.log('haga');
      resultsRef.current.push(res);
      // setResults((prev) => {
      //   if (prev) {
      //     return [...prev, res];
      //   }
      //   return [res];
      // });
      setPrediction([]);
    } else {
    }
    // console.log({ res: resultsRef.current, startRef });
    if (
      resultsRef.current &&
      resultsRef.current.length !== 0 &&
      !startRef.current
    ) {
    }
  };

  const test = async () => {};
  return (
    <div className="flex w-full flex-grow flex-col  bg-white p-2 text-center">
      <MediapipeCamera onResult={onResults} />
      <button
        onClick={() => {
          //   setLoading(true);
          const resultsCopy = [...resultsRef.current];
          resultsRef.current = [];
          console.log({ resultsCopy });
          worker
            .startPrediction(model, resultsCopy)
            .then((webWorkerMessage) => {
              console.log({ webWorkerMessage });
            });
        }}
      >
        startttt
      </button>
    </div>
  );
};
export default Camera;
