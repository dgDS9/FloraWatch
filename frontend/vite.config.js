import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';

export default defineConfig({
  base: '/FloraWatchFrontend/',
  plugins: [react(), svgr()],
  server: {
    host: true,
    port: 5173,
  },
  build: {
    // Separate heavy libraries into their own chunks for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          motion: ['motion/react'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
});
