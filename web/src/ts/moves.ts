import { MoveView } from './types';

/**
 * Detect the primary color from a move description like "RED 7" or "RED 7 + GRN 10".
 */
function detectColor(description: string): string | null {
    const map: Record<string, string> = {
        'RED': 'red',
        'YLW': 'yellow',
        'GRN': 'green',
        'BLU': 'blue',
    };
    for (const [key, val] of Object.entries(map)) {
        if (description.startsWith(key)) return val;
    }
    return null;
}

/**
 * Check if a move description is a double (contains " + ").
 */
function isDouble(description: string): boolean {
    return description.includes(' + ');
}

/**
 * Render available moves as clickable buttons.
 */
export function renderMoves(
    containerId: string,
    moves: MoveView[],
    phase: string,
    onMove: (index: number) => void,
    onSkip: () => void
): void {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    const isPlayerTurn = phase === 'player_passive' || phase === 'player_active';

    if (!isPlayerTurn) {
        // Nothing to show during bot turns or game over with no moves
        return;
    }

    if (moves.length === 0 && phase === 'player_passive') {
        // No passive moves, show skip only
        const skipBtn = createButton('Skip (no moves)', 'skip', () => onSkip());
        const group = document.createElement('div');
        group.className = 'move-group';
        group.appendChild(skipBtn);
        container.appendChild(group);
        return;
    }

    if (moves.length === 0) {
        const msg = document.createElement('div');
        msg.className = 'no-moves-msg';
        msg.textContent = 'Waiting...';
        container.appendChild(msg);
        return;
    }

    // Separate moves into categories
    const singles: MoveView[] = [];
    const doubles: MoveView[] = [];
    let strikeMove: MoveView | null = null;

    for (const move of moves) {
        if (move.is_strike) {
            strikeMove = move;
        } else if (isDouble(move.description)) {
            doubles.push(move);
        } else {
            singles.push(move);
        }
    }

    // Singles row
    if (singles.length > 0) {
        const group = document.createElement('div');
        group.className = 'move-group';
        for (const move of singles) {
            const color = detectColor(move.description);
            const btn = createButton(move.description, 'single', () => onMove(move.index));
            if (color) btn.setAttribute('data-color', color);
            group.appendChild(btn);
        }
        container.appendChild(group);
    }

    // Doubles row
    if (doubles.length > 0) {
        const group = document.createElement('div');
        group.className = 'move-group';
        for (const move of doubles) {
            const btn = createButton(move.description, 'double', () => onMove(move.index));
            group.appendChild(btn);
        }
        container.appendChild(group);
    }

    // Strike / Skip row
    const actionGroup = document.createElement('div');
    actionGroup.className = 'move-group';

    if (phase === 'player_passive') {
        // Can skip on passive turn without penalty
        const skipBtn = createButton('Skip', 'skip', () => onSkip());
        actionGroup.appendChild(skipBtn);
    }

    if (strikeMove) {
        const btn = createButton(
            phase === 'player_active' ? 'Strike (-5)' : strikeMove.description,
            'strike',
            () => onMove(strikeMove!.index)
        );
        actionGroup.appendChild(btn);
    }

    if (actionGroup.children.length > 0) {
        container.appendChild(actionGroup);
    }
}

function createButton(
    text: string,
    type: string,
    onClick: () => void
): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.className = `move-btn ${type}`;
    btn.textContent = text;
    btn.addEventListener('click', onClick);
    return btn;
}
