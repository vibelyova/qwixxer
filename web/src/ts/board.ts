import { StateView, RowView } from './types';

/**
 * Render a Qwixx scoresheet into the given container.
 */
export function renderBoard(
    containerId: string,
    state: StateView,
    _label: string,
    _isOpponent: boolean
): void {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    // Render each row
    for (const row of state.rows) {
        const rowEl = createRow(row);
        container.appendChild(rowEl);
    }

    // Render footer (strikes + score)
    const footer = createFooter(state);
    container.appendChild(footer);
}

function createRow(row: RowView): HTMLElement {
    const rowEl = document.createElement('div');
    rowEl.className = 'score-row';

    if (row.locked) {
        rowEl.classList.add('locked-row');
    }

    const freeIndex = row.free !== null
        ? row.numbers.indexOf(row.free)
        : row.numbers.length; // all crossed if null and not locked, or locked

    for (let i = 0; i < row.numbers.length; i++) {
        const num = row.numbers[i];
        const cell = document.createElement('div');
        cell.className = `cell ${row.color}`;
        cell.textContent = String(num);

        const isLastNumber = i === row.numbers.length - 1;

        if (row.locked) {
            // Entire row is locked -- show all cells dimmed, marked ones slightly brighter
            if (i < freeIndex) {
                cell.classList.add('marked');
            }
        } else if (row.free === null) {
            // Row has no free value (completely locked or fully crossed)
            cell.classList.add('past');
        } else {
            if (i < freeIndex) {
                // This number has been passed/marked
                cell.classList.add('marked');
            } else if (i === freeIndex) {
                // This is the next free number
                cell.classList.add('available');
            } else {
                // Future numbers
                cell.classList.add('available');
            }
        }

        // Lock cell treatment (last number: 12 for ascending, 2 for descending)
        if (isLastNumber && !row.locked) {
            cell.classList.add('lock-cell');
            if (row.total < 5) {
                cell.classList.add('lock-unavailable');
            }
        }

        rowEl.appendChild(cell);
    }

    // Row info: mark count
    const info = document.createElement('div');
    info.className = 'row-info';
    const count = document.createElement('span');
    count.className = 'row-count';
    count.textContent = `${row.total}x`;
    info.appendChild(count);
    rowEl.appendChild(info);

    return rowEl;
}

function createFooter(state: StateView): HTMLElement {
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
        }
        strikesArea.appendChild(box);
    }

    // Score
    const scoreEl = document.createElement('div');
    scoreEl.className = 'total-score';
    const label = document.createElement('span');
    label.textContent = 'Score ';
    scoreEl.appendChild(label);
    scoreEl.appendChild(document.createTextNode(String(state.score)));

    footer.appendChild(strikesArea);
    footer.appendChild(scoreEl);

    return footer;
}
