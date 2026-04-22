#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt

iterations = []
scores = []
winrates = []

with open('dqn_model/training_scores.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        iterations.append(int(row['iteration']))
        scores.append(float(row['avg_score']))
        if 'winrate' in row and row['winrate']:
            winrates.append(float(row['winrate']))

has_winrate = len(winrates) == len(iterations)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(iterations, scores, 'b-o', linewidth=1.5, markersize=3, label='avg score')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Average Score', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)

if has_winrate:
    ax2 = ax1.twinx()
    ax2.plot(iterations, winrates, 'r-s', linewidth=1.5, markersize=3, label='winrate (%)')
    ax2.set_ylabel('Winrate vs GA (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

plt.title('DQN Self-Play Training')
fig.tight_layout()
plt.savefig('dqn_training_plot.png', dpi=150)

summary = f'Plot saved. {len(iterations)} iterations, final avg score: {scores[-1]:.1f}'
if has_winrate:
    summary += f', final winrate: {winrates[-1]:.1f}% (peak {max(winrates):.1f}% @ iter {iterations[winrates.index(max(winrates))]})'
print(summary)
