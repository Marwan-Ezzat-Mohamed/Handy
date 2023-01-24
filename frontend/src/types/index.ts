//user can be undefined

export type User =
  | {
      id: number;
      name: string;
      email: string;
    }
  | undefined;
