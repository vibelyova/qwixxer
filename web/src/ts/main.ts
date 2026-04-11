import init, { WebGame } from '../../pkg/qwixxer_web.js';
import { GameView, MoveView, SelectionState } from './types';
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
let strikeSelected = false;

function getSelectedBot(): string {
    const select = document.getElementById('bot-select') as HTMLSelectElement;
    return select.value;
}

function findStrikeMove(moves: MoveView[]): MoveView | undefined {
    return moves.find(m => m.is_strike);
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
        strikeSelected = false;
    }

    // Get clickable and selected cell sets
    const clickableCells = isPlayerTurn && !view.game_over && !strikeSelected
        ? getClickableCellsForSelection(parsedMoves, selection)
        : new Set<string>();

    const selectedCells = new Set(selection.selected.map(t => targetKey(t)));

    // Can strike? Only on active turn and a strike move exists
    const canStrike = isPlayerTurn && !view.game_over &&
        view.phase === 'player_active' &&
        !!findStrikeMove(view.available_moves);

    // Render player board (interactive)
    renderBoard(
        'player-board',
        view.player,
        'You',
        false,
        clickableCells,
        selectedCells,
        (row, num) => onCellClick(row, num, view),
        canStrike,
        strikeSelected,
        () => onStrikeToggle(view)
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
            strikeSelected = false;
            render();
        },
        strikeSelected
    );
}

function onCellClick(row: number, num: number, view: GameView): void {
    // If strike is selected, deselect it when clicking a cell
    if (strikeSelected) {
        strikeSelected = false;
    }

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

function onStrikeToggle(view: GameView): void {
    if (strikeSelected) {
        // Deselect strike
        strikeSelected = false;
    } else {
        // Select strike, clear cell selections
        strikeSelected = true;
        const parsedMoves = view.available_moves.map(m => parseMove(m));
        selection = emptySelection(view.phase, parsedMoves);

        // Auto-confirm the strike after a brief visual flash
        const strikeMove = findStrikeMove(view.available_moves);
        if (strikeMove) {
            render(); // show the selected state briefly
            setTimeout(() => {
                handleStrike(strikeMove.index);
            }, 300);
            return;
        }
    }
    render();
}

function handleConfirm(moveIndex: number): void {
    if (!game) return;
    game.make_move(moveIndex);
    selection = { selected: [], compatibleMoves: [], phase: '' };
    strikeSelected = false;
    render();
}

function handleStrike(moveIndex: number): void {
    if (!game) return;
    game.make_move(moveIndex);
    selection = { selected: [], compatibleMoves: [], phase: '' };
    strikeSelected = false;
    render();
}

function handleSkip(): void {
    if (!game) return;
    game.skip();
    selection = { selected: [], compatibleMoves: [], phase: '' };
    strikeSelected = false;
    render();
}

function startNewGame(): void {
    if (game) {
        game.new_game(getSelectedBot());
    }
    selection = { selected: [], compatibleMoves: [], phase: '' };
    strikeSelected = false;
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
