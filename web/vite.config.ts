import { defineConfig } from 'vite';

export default defineConfig({
    root: '.',
    base: '/qwixxer/',
    build: {
        outDir: 'dist',
        assetsInlineLimit: 0,
    },
    server: {
        fs: {
            allow: ['.']
        }
    }
});
