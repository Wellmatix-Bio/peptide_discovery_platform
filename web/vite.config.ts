import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// In development the accounts proxy runs on 8081 (the job API itself takes 8080; see
// services/accounts/README.md). The app only ever calls same-origin paths, so production
// (nginx) and development (this proxy) look identical to the code and no CORS policy is
// needed anywhere.
const ACCOUNTS = process.env.ACCOUNTS_URL ?? "http://127.0.0.1:8081";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: ACCOUNTS, changeOrigin: true },
      "/auth": { target: ACCOUNTS, changeOrigin: true },
      "/healthz": { target: ACCOUNTS, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
