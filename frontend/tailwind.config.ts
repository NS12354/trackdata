import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0b0e13",
        panel: "#141a22",
        panel2: "#1b2330",
        border: "#27313f",
        muted: "#8b97a7",
        text: "#e6edf3",
        accent: "#3b82f6",
        ok: "#22c55e",
        warn: "#f59e0b",
        danger: "#ef4444",
      },
    },
  },
  plugins: [],
};

export default config;
