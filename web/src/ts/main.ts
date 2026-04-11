import init, { WebGame } from '../../pkg/qwixxer_web.js';
import { GameView } from './types';
import { renderBoard } from './board';
import { renderDice } from './dice';
import { renderMoves } from './moves';
import './style.css';

let game: WebGame | null = null;

function getSelectedBot(): string {
    const select = document.getElementById('bot-select') as HTMLSelectElement;
    return select.value;
}

function render(): void {
    if (!game) return;

    const viewJson = game.view();
    const view: GameView = JSON.parse(viewJson);

    // Render bot board
    renderBoard('bot-board', view.bot, 'Bot', true);

    // Render player board
    renderBoard('player-board', view.player, 'You', false);

    // Render dice
    renderDice('dice-container', view.dice, view.white_sum);

    // Render message
    const msgEl = document.getElementById('message');
    if (msgEl) {
        msgEl.textContent = view.message;
        msgEl.className = view.game_over ? 'game-over' : '';
    }

    // Render moves
    renderMoves(
        'moves-container',
        view.available_moves,
        view.phase,
        handleMove,
        handleSkip
    );
}

function handleMove(index: number): void {
    if (!game) return;
    game.make_move(index);
    render();
}

function handleSkip(): void {
    if (!game) return;
    game.skip();
    render();
}

function startNewGame(): void {
    if (game) {
        game.new_game(getSelectedBot());
    }
    render();
}

async function main(): Promise<void> {
    // Initialize WASM module
    await init();

    // Create game
    game = new WebGame(getSelectedBot());

    // Wire up new game button
    const newGameBtn = document.getElementById('new-game-btn');
    if (newGameBtn) {
        newGameBtn.addEventListener('click', startNewGame);
    }

    // Wire up bot selector change
    const botSelect = document.getElementById('bot-select');
    if (botSelect) {
        botSelect.addEventListener('change', startNewGame);
    }

    // Initial render
    render();
}

main().catch((err) => {
    console.error('Failed to initialize game:', err);
    const app = document.getElementById('app');
    if (app) {
        app.innerHTML = `<div style="padding: 2rem; text-align: center; color: #ef4444;">
            <h2>Failed to load game</h2>
            <p>${err}</p>
        </div>`;
    }
});
