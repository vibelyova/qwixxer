import init, { WebGame } from '../../pkg/qwixxer_web.js';
import { GameView, SelectionState } from './types';
import { renderBoard, renderScoringReference, targetKey } from './board';
import { renderDice } from './dice';
import {
    parseMove,
    getClickableCellsForSelection,
    handleCellClick,
    emptySelection,
    renderActionButtons,
} from './moves';
import './style.css';

let game: WebGame | null = null;
let selection: SelectionState = {
    selected: [],
    compatibleMoves: [],
    phase: '',
};

function getSelectedBot(): string {
    const select = document.getElementById('bot-select') as HTMLSelectElement;
    return select.value;
}

function render(): void {
    if (!game) return;

    const viewJson = game.view();
    const view: GameView = JSON.parse(viewJson);

    // Parse all moves
    const parsedMoves = view.available_moves.map(m => parseMove(m));

    // Determine if it's the player's turn
    const isPlayerTurn = view.phase === 'player_passive' || view.phase === 'player_active';

    // Reset selection if phase changed
    if (selection.phase !== view.phase) {
        selection = emptySelection(view.phase, parsedMoves);
    }

    // Get clickable and selected cell sets
    const clickableCells = isPlayerTurn && !view.game_over
        ? getClickableCellsForSelection(parsedMoves, selection)
        : new Set<string>();

    const selectedCells = new Set(selection.selected.map(t => targetKey(t)));

    // Render player board (interactive)
    renderBoard(
        'player-board',
        view.player,
        'You',
        false,
        clickableCells,
        selectedCells,
        (row, num) => onCellClick(row, num, view)
    );

    // Render bot board (non-interactive)
    renderBoard('bot-board', view.bot, 'Bot', true);

    // Render dice
    renderDice('dice-container', view.dice, view.white_sum);

    // Render message
    const msgEl = document.getElementById('message');
    if (msgEl) {
        msgEl.textContent = view.message;
        msgEl.className = view.game_over ? 'game-over' : '';
    }

    // Render action buttons
    renderActionButtons(
        'action-buttons',
        view.phase,
        selection,
        parsedMoves,
        view.available_moves,
        handleConfirm,
        handleStrike,
        handleSkip,
        () => {
            selection = emptySelection(view.phase, parsedMoves);
            render();
        }
    );
}

function onCellClick(row: number, num: number, view: GameView): void {
    const parsedMoves = view.available_moves.map(m => parseMove(m));

    const result = handleCellClick(row, num, parsedMoves, selection);
    selection = result.newSelection;

    if (result.autoConfirmMove !== null) {
        // Auto-confirm: single move that matches exactly
        handleConfirm(result.autoConfirmMove);
        return;
    }

    render();
}

function handleConfirm(moveIndex: number): void {
    if (!game) return;
    game.make_move(moveIndex);
    selection = { selected: [], compatibleMoves: [], phase: '' };
    render();
}

function handleStrike(moveIndex: number): void {
    if (!game) return;
    game.make_move(moveIndex);
    selection = { selected: [], compatibleMoves: [], phase: '' };
    render();
}

function handleSkip(): void {
    if (!game) return;
    game.skip();
    selection = { selected: [], compatibleMoves: [], phase: '' };
    render();
}

function startNewGame(): void {
    if (game) {
        game.new_game(getSelectedBot());
    }
    selection = { selected: [], compatibleMoves: [], phase: '' };
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

    // Render scoring reference (static)
    renderScoringReference('scoring-reference');

    // Initial render
    render();
}

main().catch((err) => {
    console.error('Failed to initialize game:', err);
    const app = document.getElementById('app');
    if (app) {
        app.innerHTML = `<div style="padding: 2rem; text-align: center; color: #e74c3c;">
            <h2>Failed to load game</h2>
            <p>${err}</p>
        </div>`;
    }
});
