import { MoveView, MoveTarget, ParsedMove, SelectionState } from './types';
import { targetKey } from './board';

/**
 * Color name abbreviation to row index mapping.
 */
const COLOR_TO_ROW: Record<string, number> = {
    'RED': 0,
    'YLW': 1,
    'GRN': 2,
    'BLU': 3,
};

/**
 * Parse a move description into structured targets.
 *
 * Examples:
 *   "RED 7"         -> [{ row: 0, number: 7 }]
 *   "RED 7 + GRN 10" -> [{ row: 0, number: 7 }, { row: 2, number: 10 }]
 *   "Strike"        -> [] (isStrike = true)
 */
export function parseMove(move: MoveView): ParsedMove {
    if (move.is_strike) {
        return { moveIndex: move.index, targets: [], isStrike: true, isPass: false };
    }
    if (move.is_pass) {
        return { moveIndex: move.index, targets: [], isStrike: false, isPass: true };
    }

    const targets: MoveTarget[] = [];
    const parts = move.description.split(' + ');

    for (const part of parts) {
        const tokens = part.trim().split(' ');
        if (tokens.length >= 2) {
            const colorKey = tokens[0];
            const number = parseInt(tokens[1], 10);
            const row = COLOR_TO_ROW[colorKey];
            if (row !== undefined && !isNaN(number)) {
                targets.push({ row, number });
            }
        }
    }

    return { moveIndex: move.index, targets, isStrike: false, isPass: false };
}

/**
 * Get the set of all clickable cell keys from the available moves.
 * A cell is clickable if any non-strike, non-pass move targets it.
 */
export function getClickableCells(parsedMoves: ParsedMove[]): Set<string> {
    const clickable = new Set<string>();
    for (const pm of parsedMoves) {
        if (pm.isStrike || pm.isPass) continue;
        for (const t of pm.targets) {
            clickable.add(targetKey(t));
        }
    }
    return clickable;
}

/**
 * Get the set of clickable cells given current selection state.
 * After selecting one cell, only cells from compatible moves remain clickable.
 */
export function getClickableCellsForSelection(
    parsedMoves: ParsedMove[],
    selection: SelectionState
): Set<string> {
    if (selection.selected.length === 0) {
        return getClickableCells(parsedMoves);
    }

    // Filter to moves compatible with current selection
    const compatible = parsedMoves.filter(pm =>
        selection.compatibleMoves.includes(pm.moveIndex)
    );

    const clickable = new Set<string>();
    const selectedKeys = new Set(selection.selected.map(t => targetKey(t)));

    for (const pm of compatible) {
        for (const t of pm.targets) {
            const key = targetKey(t);
            if (!selectedKeys.has(key)) {
                clickable.add(key);
            }
        }
    }

    return clickable;
}

/**
 * Given a cell click (row, number), update the selection state.
 * Returns the new selection state and optionally a move index to auto-confirm.
 */
export function handleCellClick(
    row: number,
    number: number,
    parsedMoves: ParsedMove[],
    currentSelection: SelectionState
): { newSelection: SelectionState; autoConfirmMove: number | null } {
    const clickedTarget: MoveTarget = { row, number };
    const clickedKey = targetKey(clickedTarget);

    // If clicking an already-selected cell, deselect it
    const existingIdx = currentSelection.selected.findIndex(
        t => targetKey(t) === clickedKey
    );
    if (existingIdx !== -1) {
        const newSelected = [...currentSelection.selected];
        newSelected.splice(existingIdx, 1);

        if (newSelected.length === 0) {
            return {
                newSelection: {
                    selected: [],
                    compatibleMoves: parsedMoves
                        .filter(pm => !pm.isStrike && !pm.isPass)
                        .map(pm => pm.moveIndex),
                    phase: currentSelection.phase,
                },
                autoConfirmMove: null,
            };
        }

        // Recompute compatible moves for remaining selection
        const compatible = findCompatibleMoves(newSelected, parsedMoves);
        return {
            newSelection: {
                selected: newSelected,
                compatibleMoves: compatible,
                phase: currentSelection.phase,
            },
            autoConfirmMove: null,
        };
    }

    // Add the clicked cell to the selection
    const newSelected = [...currentSelection.selected, clickedTarget];

    // Find moves that match ALL selected targets
    const compatible = findCompatibleMoves(newSelected, parsedMoves);

    if (compatible.length === 0) {
        // No moves match this combination -- ignore the click
        return { newSelection: currentSelection, autoConfirmMove: null };
    }

    // Never auto-confirm — always require explicit confirm button
    return {
        newSelection: {
            selected: newSelected,
            compatibleMoves: compatible,
            phase: currentSelection.phase,
        },
        autoConfirmMove: null,
    };
}

/**
 * Find move indices whose targets are a superset of the given selected targets.
 */
function findCompatibleMoves(
    selected: MoveTarget[],
    parsedMoves: ParsedMove[]
): number[] {
    const selectedKeys = new Set(selected.map(t => targetKey(t)));

    return parsedMoves
        .filter(pm => {
            if (pm.isStrike || pm.isPass) return false;
            // Every selected target must appear in this move's targets
            for (const key of selectedKeys) {
                if (!pm.targets.some(t => targetKey(t) === key)) {
                    return false;
                }
            }
            // The move must have at least as many targets as what's selected
            return pm.targets.length >= selected.length;
        })
        .map(pm => pm.moveIndex);
}

/**
 * Find the unique move that exactly matches the current selection.
 * Returns the move index or null if ambiguous/none.
 */
export function findExactMove(
    selection: SelectionState,
    parsedMoves: ParsedMove[]
): number | null {
    const selectedKeys = new Set(selection.selected.map(t => targetKey(t)));

    const exactMatches = parsedMoves.filter(pm => {
        if (pm.isStrike || pm.isPass) return false;
        if (pm.targets.length !== selection.selected.length) return false;
        return pm.targets.every(t => selectedKeys.has(targetKey(t)));
    });

    if (exactMatches.length >= 1) {
        return exactMatches[0].moveIndex;
    }

    return null;
}

/**
 * Create the initial (empty) selection state.
 */
export function emptySelection(phase: string, parsedMoves: ParsedMove[]): SelectionState {
    return {
        selected: [],
        compatibleMoves: parsedMoves
            .filter(pm => !pm.isStrike && !pm.isPass)
            .map(pm => pm.moveIndex),
        phase,
    };
}

/**
 * Render the action buttons (Confirm, Skip).
 * Strike is now handled via clickable strike boxes on the scoresheet.
 * Clear is removed; clicking a selected cell deselects it.
 */
export function renderActionButtons(
    containerId: string,
    phase: string,
    selection: SelectionState,
    parsedMoves: ParsedMove[],
    _moves: MoveView[],
    onConfirm: (moveIndex: number) => void,
    _onStrike: (moveIndex: number) => void,
    onSkip: () => void,
    _onReset: () => void,
    strikeSelected?: boolean
): void {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    const isPlayerTurn = phase === 'player_passive' || phase === 'player_active';
    if (!isPlayerTurn) return;

    // Check if there is a confirmable selection
    const exactMove = findExactMove(selection, parsedMoves);
    const hasSelection = selection.selected.length > 0;

    // Confirm button (for cell selections or strike)
    if (hasSelection || strikeSelected) {
        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'action-btn confirm';
        confirmBtn.textContent = 'Confirm';

        if (strikeSelected) {
            const strikeMove = _moves.find(m => m.is_strike);
            if (strikeMove) {
                confirmBtn.addEventListener('click', () => _onStrike(strikeMove.index));
            }
        } else if (exactMove !== null) {
            confirmBtn.addEventListener('click', () => onConfirm(exactMove));
        } else {
            confirmBtn.disabled = true;
            confirmBtn.title = 'Select all required squares first';
        }
        container.appendChild(confirmBtn);
    }

    // Skip button (passive turn only)
    if (phase === 'player_passive') {
        const skipBtn = document.createElement('button');
        skipBtn.className = 'action-btn skip';
        skipBtn.textContent = 'Skip';
        skipBtn.addEventListener('click', onSkip);
        container.appendChild(skipBtn);
    }
}
