import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const allowedHosts = ["localhost", "0.0.0.0", "tetris.cgi.lab.nycu.edu.tw"]

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // dev server (`npm run dev`)
  server: {
    host: "0.0.0.0",
    allowedHosts,
  },
  // production preview of the built dist/ (`npm run preview`) -- this is what
  // nginx proxies to on :62200 for the real demo. Note: vite preview reads this
  // block, NOT `server`, so host/allowedHosts must be repeated here.
  preview: {
    host: "0.0.0.0",
    port: 62200,
    allowedHosts,
  },
})
