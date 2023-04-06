/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        gold: "#FEB86F",
        violet: "#9194CE",
        red: "#F492A0",
        blue: "#3E479B",
      },
      screens: {
        xs: "475px",
      },
      gridTemplateColumns: {
        fluid: "repeat(auto-fit, minmax(20rem, 1fr))",
      },
    },
    screens: {
      sm: "375px",
      // => @media (min-width: 640px) { ... }

      md: "768px",
      // => @media (min-width: 768px) { ... }

      lg: "1024px",
      // => @media (min-width: 1024px) { ... }

      xl: "1280px",
      // => @media (min-width: 1280px) { ... }

      "2xl": "1536px",
      // => @media (min-width: 1536px) { ... }
    },
    plugins: [],
    safelist: [
      {
        pattern: /^bg-/,
      },
      {
        pattern: /^text-/,
      },
      {
        pattern: /^border-/,
      },
    ],
  },
  plugins: [require("daisyui")],
};
