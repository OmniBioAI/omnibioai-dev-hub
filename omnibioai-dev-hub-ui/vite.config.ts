import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    base: "/_svc/devhub",
    plugins: [react()],
    define: {
      __OBAI_STUDIO_ORIGIN__: JSON.stringify(env.VITE_STUDIO_ORIGIN || ""),
    },

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
  };
});
