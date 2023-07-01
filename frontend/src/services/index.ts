const getLessons = async () => {
  return fetch("Handy.json").then((res) => res.json());
};

export { getLessons };
