import create from "zustand";

import { persist, devtools } from "zustand/middleware";
import { UserSlice, createUserSlice } from "./slices/userSlice";

export type Store = UserSlice;

export const useGlobalStore = create<Store>()(
  devtools(
    persist(
      (...a) => ({
        ...createUserSlice(...a),
      }),
      {
        name: "data", // name of item in the storage (must be unique)
        partialize: (state) => ({
          user: state.user,
        }), // only persist the customer slice
      }
    )
  )
);
export default useGlobalStore;
