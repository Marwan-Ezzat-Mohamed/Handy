import React from "react";

import useGlobalStore from "./stores/zustand";
import { shallow } from "zustand/shallow";

function App() {
  const { user, setUser } = useGlobalStore(
    (state) => ({ user: state.user, setUser: state.setUser }),
    shallow
  );

  return (
    <div className="App">
      {JSON.stringify(user)}

      <button onClick={() => setUser({ id: 23, name: "John", email: "email" })}>
        Set User
      </button>
    </div>
  );
}

export default App;
