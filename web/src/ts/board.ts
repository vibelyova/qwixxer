import { StateView, RowView, MoveTarget } from './types';

/** Score lookup: how many marks -> points */
const SCORE_TABLE: number[] = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78];

/** Callback type for when a square is clicked */
export type CellClickHandler = (row: number, number: number) => void;

/** Callback type for when a strike box is clicked */
export type StrikeClickHandler = () => void;

/**
 * Render a Qwixx scoresheet into the given container.
 *
 * @param containerId - DOM element id
 * @param state - the player/bot state view
 * @param isOpponent - if true, non-interactive (smaller, dimmed)
 * @param clickableCells - set of "row:number" keys for cells that should be clickable
 * @param selectedCells - set of "row:number" keys for cells that are currently selected
 * @param onCellClick - callback when a clickable cell is clicked
 * @param canStrike - whether the strike boxes should be clickable
 * @param strikeSelected - whether a strike is currently selected (toggled)
 * @param onStrikeClick - callback when a strike box is clicked
 */
export function renderBoard(
    containerId: string,
    state: StateView,
    _label: string,
    isOpponent: boolean,
    clickableCells?: Set<string>,
    selectedCells?: Set<string>,
    onCellClick?: CellClickHandler,
    canStrike?: boolean,
    strikeSelected?: boolean,
    onStrikeClick?: StrikeClickHandler
): void {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    // Render each row
    for (let rowIndex = 0; rowIndex < state.rows.length; rowIndex++) {
        const row = state.rows[rowIndex];
        const rowEl = createRow(
            row,
            rowIndex,
            isOpponent,
            clickableCells,
            selectedCells,
            onCellClick
        );
        container.appendChild(rowEl);
    }

    // Render footer (strikes + totals)
    const footer = createFooter(state, isOpponent, canStrike, strikeSelected, onStrikeClick);
    container.appendChild(footer);
}

function createRow(
    row: RowView,
    rowIndex: number,
    isOpponent: boolean,
    clickableCells?: Set<string>,
    selectedCells?: Set<string>,
    onCellClick?: CellClickHandler
): HTMLElement {
    const rowEl = document.createElement('div');
    rowEl.className = 'score-row';

    if (row.locked) {
        rowEl.classList.add('locked-row');
    }

    // Determine the free boundary index for "available" highlight
    const freeIndex = row.free !== null
        ? row.numbers.indexOf(row.free)
        : row.numbers.length;

    for (let i = 0; i < row.numbers.length; i++) {
        const num = row.numbers[i];
        const cell = document.createElement('div');
        cell.className = `cell ${row.color}`;

        const isLastNumber = i === row.numbers.length - 1;
        const cellKey = `${rowIndex}:${num}`;
        const isMarked = row.marks[i];

        // Cell content: number
        const numberSpan = document.createElement('span');
        numberSpan.className = 'cell-number';
        numberSpan.textContent = String(num);
        cell.appendChild(numberSpan);

        // State classes based on marks tracking
        if (isMarked) {
            // This specific number was marked
            cell.classList.add('marked');
            // Add X overlay
            const xSpan = document.createElement('span');
            xSpan.className = 'cell-x';
            xSpan.textContent = 'X';
            cell.appendChild(xSpan);
        } else if (row.locked) {
            // Row is locked but this cell wasn't marked - show as skipped
            cell.classList.add('skipped');
        } else if (row.free === null) {
            // Row is somehow past without lock
            cell.classList.add('skipped');
        } else if (i < freeIndex) {
            // Before the free pointer but NOT marked = skipped/blank
            cell.classList.add('skipped');
        } else {
            // At or after free pointer = available
            cell.classList.add('available');
        }

        // Lock cell treatment (last number: 12 for ascending, 2 for descending)
        if (isLastNumber && !row.locked && row.total < 5 && !isMarked) {
            cell.classList.add('lock-cell');
            cell.classList.add('lock-unavailable');
        }

        // Interactive states (only for player board)
        const isClickable = !isOpponent && clickableCells && clickableCells.has(cellKey);
        const isSelected = !isOpponent && selectedCells && selectedCells.has(cellKey);

        if (isClickable || isSelected) {
            cell.classList.add('clickable');
            cell.addEventListener('click', () => {
                if (onCellClick) {
                    onCellClick(rowIndex, num);
                }
            });
        }

        if (isSelected) {
            cell.classList.add('selected');
        }

        rowEl.appendChild(cell);
    }

    // Lock label
    if (!isOpponent) {
        const lockLabel = document.createElement('div');
        lockLabel.className = 'lock-label';
        lockLabel.textContent = 'Min 5\u00D7';
        rowEl.appendChild(lockLabel);
    }

    // Row info: mark count
    const info = document.createElement('div');
    info.className = 'row-info';
    const count = document.createElement('span');
    count.className = 'row-count';
    count.textContent = `${row.total}\u00D7`;
    info.appendChild(count);
    rowEl.appendChild(info);

    return rowEl;
}

function createFooter(
    state: StateView,
    isOpponent: boolean,
    canStrike?: boolean,
    strikeSelected?: boolean,
    onStrikeClick?: StrikeClickHandler
): HTMLElement {
    const footer = document.createElement('div');
    footer.className = 'sheet-footer';

    // Strikes
    const strikesArea = document.createElement('div');
    strikesArea.className = 'strikes-area';

    const strikesLabel = document.createElement('span');
    strikesLabel.className = 'strikes-label';
    strikesLabel.textContent = 'Strikes';
    strikesArea.appendChild(strikesLabel);

    for (let i = 0; i < 4; i++) {
        const box = document.createElement('div');
        box.className = 'strike-box';
        if (i < state.strikes) {
            box.classList.add('struck');
            box.textContent = 'X';
        } else if (!isOpponent && canStrike && i === state.strikes) {
            // The next empty strike box is clickable
            box.classList.add('strike-clickable');
            if (strikeSelected) {
                box.classList.add('strike-selected');
                box.textContent = 'X';
            }
            box.addEventListener('click', () => {
                if (onStrikeClick) onStrikeClick();
            });
        }
        strikesArea.appendChild(box);
    }

    const penalty = document.createElement('span');
    penalty.className = 'strikes-penalty';
    penalty.textContent = 'X = -5';
    strikesArea.appendChild(penalty);

    footer.appendChild(strikesArea);

    // If it's the player's board, show detailed totals
    if (!isOpponent) {
        const totalsRow = document.createElement('div');
        totalsRow.className = 'totals-row';

        const colors = ['red', 'yellow', 'green', 'blue'];
        const rowScores: number[] = [];

        for (let i = 0; i < state.rows.length; i++) {
            const markCount = state.rows[i].total;
            const rowScore = SCORE_TABLE[markCount] || 0;
            rowScores.push(rowScore);

            const cell = document.createElement('div');
            cell.className = `total-cell ${colors[i]}`;
            cell.textContent = String(rowScore);
            totalsRow.appendChild(cell);

            if (i < state.rows.length - 1) {
                const op = document.createElement('span');
                op.className = 'total-operator';
                op.textContent = '+';
                totalsRow.appendChild(op);
            }
        }

        // Minus strikes
        const minusOp = document.createElement('span');
        minusOp.className = 'total-operator';
        minusOp.textContent = '\u2212';
        totalsRow.appendChild(minusOp);

        const strikePenalty = state.strikes * 5;
        const strikeCell = document.createElement('div');
        strikeCell.className = 'total-cell strikes';
        strikeCell.textContent = String(strikePenalty);
        totalsRow.appendChild(strikeCell);

        // Equals total
        const eqOp = document.createElement('span');
        eqOp.className = 'total-operator';
        eqOp.textContent = '=';
        totalsRow.appendChild(eqOp);

        const totalCell = document.createElement('div');
        totalCell.className = 'total-cell final-total';
        totalCell.textContent = String(state.score);
        totalsRow.appendChild(totalCell);

        footer.appendChild(totalsRow);
    } else {
        // Simple score for opponent
        const scoreEl = document.createElement('div');
        scoreEl.style.fontWeight = '800';
        scoreEl.style.fontSize = '0.9rem';
        scoreEl.textContent = `Score: ${state.score}`;
        footer.appendChild(scoreEl);
    }

    return footer;
}

/**
 * Render the scoring reference table at the bottom.
 */
export function renderScoringReference(containerId: string): void {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    const table = document.createElement('table');
    table.className = 'scoring-table';

    const headerRow = document.createElement('tr');
    const bodyRow = document.createElement('tr');

    const marks = ['1\u00D7', '2\u00D7', '3\u00D7', '4\u00D7', '5\u00D7', '6\u00D7',
                   '7\u00D7', '8\u00D7', '9\u00D7', '10\u00D7', '11\u00D7', '12\u00D7'];
    const points = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78];

    for (let i = 0; i < marks.length; i++) {
        const th = document.createElement('th');
        th.textContent = marks[i];
        headerRow.appendChild(th);

        const td = document.createElement('td');
        td.textContent = String(points[i]);
        bodyRow.appendChild(td);
    }

    table.appendChild(headerRow);
    table.appendChild(bodyRow);
    container.appendChild(table);
}

/**
 * Build a set key from a move target.
 */
export function targetKey(target: MoveTarget): string {
    return `${target.row}:${target.number}`;
}
