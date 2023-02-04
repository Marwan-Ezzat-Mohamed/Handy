import React, { useEffect, useState } from "react";
import { Results, NormalizedLandmark } from "@mediapipe/holistic";
import * as tf from "@tensorflow/tfjs";

import { modelAtom } from "./stores/jotai";
import { useAtom } from "jotai";
import { MediapipeCamera } from "./components/MediapipeCamera";
import { start } from "repl";
import { labelMap } from "./utils/index";

function App() {
  const [model, setModel] = useAtom(modelAtom);
  const startRef = React.useRef(false);
  const [results, setResults] = useState<null | Results[]>([]);
  const [prediction, setPrediction] = useState<Array<string>>([]);
  const [predict, setPredict] = useState(false);

  function chunkArray(arr: any[], chunkSize: number): any[][] {
    let result = [];
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

  useEffect(() => {
    async function load() {
      const model = await tf.loadLayersModel("./model.json");
      setModel(model);
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    startPrediction();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predict]);

  async function startPrediction() {
    if (results && results.length >= 15) {
      const chunks = chunkArray(results, 15);

      const sequences = chunks
        .map((chunk) => {
          return chunk.map((result) => extractKeypoints(result));
        })
        .filter((seq) => seq.length === 15);

      console.log(chunks);

      const res: any = {
        label: null,
        confidence: null,
      };
      //use the layers model to predict the gesture and get the confidence score and the label
      if (model) {
        const prediction = model.predict(tf.tensor3d(sequences)) as tf.Tensor;
        const data = await prediction.data();

        const labels = prediction.argMax(-1).dataSync();
        const confidences = prediction.softmax().dataSync();

        console.log({
          labels,
          confidences,
        });

        let max = 0;
        let index = 0;
        for (let i = 0; i < data.length; i++) {
          if (data[i] > max) {
            max = data[i];
            index = i;
          }
        }
        //console.log(data);
        res.label = index;
        res.confidence = max;
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
      }
      setResults([]);
    }
  }

  const onResults = (results: Results) => {
    if (startRef.current) {
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
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        width: "100vw",
        backgroundColor: "black",
        color: "white",
      }}
    >
      <MediapipeCamera onResult={onResults} />
      <button
        onClick={() => {
          startRef.current = !startRef.current;
        }}
      >
        {startRef.current ? "Stop" : "Start"}
      </button>
      <label
        style={{
          fontSize: 50,
        }}
      >
        {prediction.filter((res) => res !== "africa").join(" ")}
      </label>
    </div>
  );
}

export default App;
