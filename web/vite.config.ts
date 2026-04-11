import { defineConfig } from 'vite';

export default defineConfig({
    root: '.',
    base: '/qwixxer/',
    build: {
        outDir: 'dist',
        assetsInlineLimit: 0,
        rollupOptions: {
            input: {
                main: 'index.html',
                explorer: 'explorer.html',
            },
        },
    },
    server: {
        fs: {
            allow: ['.']
        }
    }
});
