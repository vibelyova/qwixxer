const DIE_COLORS: string[] = ['white', 'white', 'red', 'yellow', 'green', 'blue'];
const DIE_LABELS: string[] = ['W1', 'W2', 'Red', 'Ylw', 'Grn', 'Blu'];

/**
 * Render dice display with colored dice and white sum.
 */
export function renderDice(
    containerId: string,
    dice: number[],
    whiteSum: number
): void {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    // Individual dice
    for (let i = 0; i < dice.length; i++) {
        const die = document.createElement('div');
        die.className = `die ${DIE_COLORS[i]}`;
        die.textContent = String(dice[i]);

        const label = document.createElement('span');
        label.className = 'die-label';
        label.textContent = DIE_LABELS[i];
        die.appendChild(label);

        container.appendChild(die);
    }

    // White sum indicator
    const sumBox = document.createElement('div');
    sumBox.className = 'white-sum';

    const sumLabel = document.createElement('span');
    sumLabel.className = 'white-sum-label';
    sumLabel.textContent = 'Sum';
    sumBox.appendChild(sumLabel);

    const sumValue = document.createElement('span');
    sumValue.className = 'white-sum-value';
    sumValue.textContent = String(whiteSum);
    sumBox.appendChild(sumValue);

    container.appendChild(sumBox);
}
