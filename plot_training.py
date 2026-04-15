#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt

iterations = []
scores = []

with open('dqn_model/training_scores.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        iterations.append(int(row['iteration']))
        scores.append(float(row['avg_score']))

plt.figure(figsize=(10, 5))
plt.plot(iterations, scores, 'b-', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel('Average Score')
plt.title('DQN Self-Play Training (with augmentation + 1v1)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dqn_training_plot.png', dpi=150)
print(f'Plot saved. {len(iterations)} iterations, final avg score: {scores[-1]:.1f}')
