import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/_svc/devhub",
  plugins: [react()],

  server: {
    port: 5173,

    // IMPORTANT: connect UI → FastAPI backend
    proxy: {
      // TES discovery keeps its versioned /api/discovery contract intact.
      "/api/discovery": {
        target: "http://127.0.0.1:8082",
        changeOrigin: true,
      },
      "/api": {
        target: "http://127.0.0.1:8082",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/rag": {
        target: "http://127.0.0.1:8082",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8082",
        changeOrigin: true,
      },
    },
  },
});
