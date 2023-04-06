//user can be undefined

import { Results } from "@mediapipe/holistic";

export type User =
  | {
      id: number;
      name: string;
      email: string;
    }
  | undefined;

//PickOnly "leftHandLandmarks" and "rightHandLandmarks" from Results
export type PredictType = Pick<
  Results,
  "leftHandLandmarks" | "rightHandLandmarks"
>;

const x: PredictType = {
  leftHandLandmarks: [],
  rightHandLandmarks: [],
};
