import { StateCreator } from "zustand";
import { Store } from "..";
import { getLessons } from "../../../services";
import { Lessons } from "../../../types";

export interface LessonSlice {
  lessonInformation: {
    [key: string]: {
      progress: number;
    };
  };
  updateLessonProgress: (lessonName: string, progress: number) => void;
  lessons: {
    [key: string]: Array<string>;
  };
  fetchLessons: () => Promise<void>;
}
export const createLessonSlice: StateCreator<Store, [], [], LessonSlice> = (
  set
) => ({
  lessonInformation: {},
  updateLessonProgress: (lessonName, progress) => {
    set((state) => {
      return {
        lessonInformation: {
          ...state.lessonInformation,
          [lessonName]: {
            progress,
          },
        },
      };
    });
  },
  lessons: {},
  fetchLessons: async () => {
    const lessons = await getLessons();
    //sort them by alphabetical order of the lesson name
    const sortedLessons = Object.keys(lessons)
      .sort((a, b) => a.localeCompare(b) || a.length - b.length)
      .reduce((acc: Lessons, key) => {
        acc[key] = lessons[key];
        return acc;
      }, {});
    set(() => {
      return {
        lessons: sortedLessons,
      };
    });
  },
});
