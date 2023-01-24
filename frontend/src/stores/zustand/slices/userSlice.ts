import { StateCreator } from "zustand";
import { Store } from "..";
import { User } from "../../../types";

export interface UserSlice {
  user: User;
  setUser(user: User): void;
}
export const createUserSlice: StateCreator<Store, [], [], UserSlice> = (
  set
) => ({
  user: undefined,
  setUser(user: User) {
    set({ user });
  },
});
