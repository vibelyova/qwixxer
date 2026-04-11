import init, { StateExplorer } from '../../pkg/qwixxer_web.js';
import './style.css';

// ---- Types ----

interface EvalResult {
    ga_value: number;
    dqn_value: number;
    score: number;
    blanks: number;
    probability: number;
    weighted_probability: number;
    gene_breakdown: GeneContribution[];
}

interface GeneContribution {
    name: string;
    raw_value: number;
    weight: number;
    contribution: number;
}

// ---- State ----

// marks[row][i] = whether that cell is marked
// rows 0,1 ascending: index 0 = number 2, index 10 = number 12
// rows 2,3 descending: index 0 = number 12, index 10 = number 2
const marks: boolean[][] = [
    new Array(11).fill(false),
    new Array(11).fill(false),
    new Array(11).fill(false),
    new Array(11).fill(false),
];
let strikes = 0;
let numOpponents = 1;
let maxOppStrikes = 0;
let scoreGap = 0;

let explorer: StateExplorer | null = null;

// ---- Constants ----

const ROW_COLORS = ['red', 'yellow', 'green', 'blue'];
const ROW_NUMBERS: number[][] = [
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  // red ascending
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  // yellow ascending
    [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],  // green descending
    [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],  // blue descending
];

// ---- Rendering ----

function renderBoard(): void {
    const container = document.getElementById('scoresheet');
    if (!container) return;
    container.innerHTML = '';

    for (let rowIdx = 0; rowIdx < 4; rowIdx++) {
        const rowEl = document.createElement('div');
        rowEl.className = 'score-row';

        const numbers = ROW_NUMBERS[rowIdx];
        for (let i = 0; i < numbers.length; i++) {
            const num = numbers[i];
            const cell = document.createElement('div');
            cell.className = `cell ${ROW_COLORS[rowIdx]}`;

            // Number text
            const numberSpan = document.createElement('span');
            numberSpan.className = 'cell-number';
            numberSpan.textContent = String(num);
            cell.appendChild(numberSpan);

            if (marks[rowIdx][i]) {
                cell.classList.add('marked');
                const xSpan = document.createElement('span');
                xSpan.className = 'cell-x';
                xSpan.textContent = 'X';
                cell.appendChild(xSpan);
            }

            // Lock cell indicator for the last position
            if (i === 10) {
                cell.classList.add('lock-cell');
                const lockIcon = document.createElement('span');
                lockIcon.className = 'lock-icon';
                lockIcon.textContent = '\uD83D\uDD12';
                cell.appendChild(lockIcon);
            }

            // Always clickable
            cell.classList.add('clickable');
            cell.style.cursor = 'pointer';
            const ri = rowIdx;
            const ci = i;
            cell.addEventListener('click', () => {
                marks[ri][ci] = !marks[ri][ci];
                renderBoard();
                evaluate();
            });

            rowEl.appendChild(cell);
        }

        // Lock label
        const lockLabel = document.createElement('div');
        lockLabel.className = 'lock-label';
        lockLabel.textContent = 'Min 5\u00D7';
        rowEl.appendChild(lockLabel);

        // Row count
        const info = document.createElement('div');
        info.className = 'row-info';
        const count = document.createElement('span');
        count.className = 'row-count';
        const total = marks[rowIdx].filter(m => m).length;
        count.textContent = `${total}\u00D7`;
        info.appendChild(count);
        rowEl.appendChild(info);

        container.appendChild(rowEl);
    }

    // Footer with strikes
    const footer = document.createElement('div');
    footer.className = 'sheet-footer';

    const strikesArea = document.createElement('div');
    strikesArea.className = 'strikes-area';

    const strikesLabel = document.createElement('span');
    strikesLabel.className = 'strikes-label';
    strikesLabel.textContent = 'Strikes';
    strikesArea.appendChild(strikesLabel);

    for (let i = 0; i < 4; i++) {
        const box = document.createElement('div');
        box.className = 'strike-box';
        box.style.cursor = 'pointer';

        if (i < strikes) {
            box.classList.add('struck');
            box.textContent = 'X';
        }

        box.addEventListener('click', () => {
            if (i < strikes) {
                // Clicking a struck box removes it (and all after)
                strikes = i;
            } else if (i === strikes) {
                // Clicking the next empty box adds a strike
                strikes = i + 1;
            }
            renderBoard();
            evaluate();
        });

        strikesArea.appendChild(box);
    }

    const penalty = document.createElement('span');
    penalty.className = 'strikes-penalty';
    penalty.textContent = 'X = -5';
    strikesArea.appendChild(penalty);

    footer.appendChild(strikesArea);
    container.appendChild(footer);
}

function evaluate(): void {
    if (!explorer) return;

    const stateJson = JSON.stringify({
        marks,
        strikes,
        num_opponents: numOpponents,
        max_opponent_strikes: maxOppStrikes,
        score_gap: scoreGap,
    });

    const resultJson = explorer.evaluate(stateJson);
    const result: EvalResult = JSON.parse(resultJson);

    renderEvaluation(result);
}

function renderEvaluation(result: EvalResult): void {
    const container = document.getElementById('eval-display');
    if (!container) return;
    container.innerHTML = '';

    // Main metrics grid
    const grid = document.createElement('div');
    grid.className = 'eval-grid';

    const metrics: [string, string][] = [
        ['Score', String(result.score)],
        ['Blanks', String(result.blanks)],
        ['Probability', (result.probability * 100).toFixed(1) + '%'],
        ['Weighted Prob', result.weighted_probability.toFixed(3)],
        ['GA Value', result.ga_value.toFixed(4)],
        ['DQN Value', result.dqn_value.toFixed(4)],
    ];

    for (const [label, value] of metrics) {
        const item = document.createElement('div');
        item.className = 'eval-item';
        const labelEl = document.createElement('div');
        labelEl.className = 'eval-label';
        labelEl.textContent = label;
        const valueEl = document.createElement('div');
        valueEl.className = 'eval-value';
        valueEl.textContent = value;
        item.appendChild(labelEl);
        item.appendChild(valueEl);
        grid.appendChild(item);
    }

    container.appendChild(grid);

    // Gene breakdown table
    if (result.gene_breakdown.length > 0) {
        const heading = document.createElement('div');
        heading.className = 'eval-section-heading';
        heading.textContent = 'GA Gene Breakdown';
        container.appendChild(heading);

        const table = document.createElement('table');
        table.className = 'gene-table';

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        for (const h of ['Gene', 'Raw', 'Weight', 'Contribution']) {
            const th = document.createElement('th');
            th.textContent = h;
            headerRow.appendChild(th);
        }
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        for (const gene of result.gene_breakdown) {
            const row = document.createElement('tr');
            const nameCell = document.createElement('td');
            nameCell.textContent = gene.name;
            nameCell.className = 'gene-name';
            row.appendChild(nameCell);

            const rawCell = document.createElement('td');
            rawCell.textContent = gene.raw_value.toFixed(3);
            rawCell.className = 'gene-num';
            row.appendChild(rawCell);

            const weightCell = document.createElement('td');
            weightCell.textContent = gene.weight.toFixed(3);
            weightCell.className = 'gene-num';
            row.appendChild(weightCell);

            const contribCell = document.createElement('td');
            contribCell.textContent = gene.contribution.toFixed(4);
            contribCell.className = 'gene-num';

            // Color contribution: green positive, red negative
            if (gene.contribution > 0) {
                contribCell.style.color = 'var(--green-dark)';
            } else if (gene.contribution < 0) {
                contribCell.style.color = 'var(--red-dark)';
            }
            row.appendChild(contribCell);

            tbody.appendChild(row);
        }

        // Total row
        const totalRow = document.createElement('tr');
        totalRow.className = 'gene-total-row';
        const totalLabel = document.createElement('td');
        totalLabel.textContent = 'Total';
        totalLabel.colSpan = 3;
        totalLabel.style.fontWeight = '800';
        totalRow.appendChild(totalLabel);
        const totalVal = document.createElement('td');
        totalVal.className = 'gene-num';
        totalVal.style.fontWeight = '800';
        const totalContrib = result.gene_breakdown.reduce((s, g) => s + g.contribution, 0);
        totalVal.textContent = totalContrib.toFixed(4);
        totalRow.appendChild(totalVal);
        tbody.appendChild(totalRow);

        table.appendChild(tbody);
        container.appendChild(table);
    }
}

// ---- Init ----

async function main(): Promise<void> {
    await init();
    explorer = new StateExplorer();

    // Wire up sliders
    const oppSlider = document.getElementById('opponents-slider') as HTMLInputElement;
    const oppValue = document.getElementById('opponents-value')!;
    oppSlider.addEventListener('input', () => {
        numOpponents = parseInt(oppSlider.value);
        oppValue.textContent = oppSlider.value;
        evaluate();
    });

    const oppStrikesSlider = document.getElementById('opp-strikes-slider') as HTMLInputElement;
    const oppStrikesValue = document.getElementById('opp-strikes-value')!;
    oppStrikesSlider.addEventListener('input', () => {
        maxOppStrikes = parseInt(oppStrikesSlider.value);
        oppStrikesValue.textContent = oppStrikesSlider.value;
        evaluate();
    });

    const gapSlider = document.getElementById('score-gap-slider') as HTMLInputElement;
    const gapValue = document.getElementById('score-gap-value')!;
    gapSlider.addEventListener('input', () => {
        scoreGap = parseInt(gapSlider.value);
        gapValue.textContent = gapSlider.value;
        evaluate();
    });

    renderBoard();
    evaluate();
}

main().catch((err) => {
    console.error('Failed to initialize explorer:', err);
    const app = document.getElementById('app');
    if (app) {
        app.innerHTML = `<div style="padding: 2rem; text-align: center; color: #e74c3c;">
            <h2>Failed to load State Explorer</h2>
            <p>${err}</p>
        </div>`;
    }
});
