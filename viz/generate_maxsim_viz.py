#!/usr/bin/env python3
"""
Generate MaxSim visualization charts using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

# 1. MaxSim Alignment: Best matches per query token
fig, ax = plt.subplots(figsize=(10, 6))

query_tokens = ['capital', 'of', 'France']
# Simulated alignment scores (query token × document token)
alignment_matrix = np.array([
    [0.2, 0.1, 0.15, 0.95, 0.3, 0.2],  # "capital" aligns best with "capital" (0.95)
    [0.1, 0.2, 0.1, 0.2, 0.85, 0.15],  # "of" aligns best with "of" (0.85)
    [0.1, 0.1, 0.1, 0.1, 0.2, 0.92],   # "France" aligns best with "France" (0.92)
])

max_scores = np.max(alignment_matrix, axis=1)
maxsim_total = np.sum(max_scores)

colors = ['#00d9ff', '#ff6b9d', '#ffd93d']
bars = ax.bar(query_tokens, max_scores, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=1.5)

ax.set_xlabel('Query Token', fontweight='bold')
ax.set_ylabel('Max Similarity Score', fontweight='bold')
ax.set_title(f'MaxSim: Best Match per Query Token\nTotal MaxSim Score: {maxsim_total:.2f}\nEach query token finds its best-matching document token', 
             fontweight='bold', pad=15)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, score in zip(bars, max_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'maxsim_alignment.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: maxsim_alignment.png")

# 2. Token Alignment Heatmap
fig, ax = plt.subplots(figsize=(12, 6))

doc_tokens = ['Paris', 'is', 'the', 'capital', 'of', 'France']
alignment_matrix_display = alignment_matrix

im = ax.imshow(alignment_matrix_display, cmap='YlOrRd', aspect='auto', 
               vmin=0, vmax=1, interpolation='nearest')

# Set ticks
ax.set_xticks(np.arange(len(doc_tokens)))
ax.set_yticks(np.arange(len(query_tokens)))
ax.set_xticklabels(doc_tokens)
ax.set_yticklabels(query_tokens)

# Add text annotations
for i in range(len(query_tokens)):
    for j in range(len(doc_tokens)):
        score = alignment_matrix_display[i, j]
        color = 'white' if score > 0.5 else 'black'
        ax.text(j, i, f'{score:.2f}', ha='center', va='center', 
                color=color, fontweight='bold', fontsize=10)

ax.set_xlabel('Document Tokens', fontweight='bold', fontsize=12)
ax.set_ylabel('Query Tokens', fontweight='bold', fontsize=12)
ax.set_title('Token Alignment Matrix: Query × Document\nDarker = higher similarity. Each query token finds max in its row.', 
             fontweight='bold', pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Similarity Score', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'maxsim_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: maxsim_heatmap.png")

# 3. Comparison: Dense vs MaxSim
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Dense (single vector)
query_vec = np.array([0.8, 0.6])
doc_vec = np.array([0.75, 0.65])
dense_score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))

ax1.arrow(0, 0, query_vec[0], query_vec[1], head_width=0.05, head_length=0.05, 
          fc='#00d9ff', ec='#00d9ff', linewidth=3, label='Query vector')
ax1.arrow(0, 0, doc_vec[0], doc_vec[1], head_width=0.05, head_length=0.05, 
          fc='#ff6b9d', ec='#ff6b9d', linewidth=3, label='Document vector')
ax1.set_xlim(-0.1, 1.0)
ax1.set_ylim(-0.1, 1.0)
ax1.set_xlabel('Dimension 1', fontweight='bold')
ax1.set_ylabel('Dimension 2', fontweight='bold')
ax1.set_title(f'Dense Embedding\nSingle vector per query/doc\nCosine: {dense_score:.3f}', 
              fontweight='bold')
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Right: MaxSim (token-level)
# Show multiple vectors
query_vectors = np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]])
doc_vectors = np.array([[0.85, 0.15], [0.15, 0.85], [0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])

colors_q = ['#00d9ff', '#00d9ff', '#00d9ff']
colors_d = ['#ff6b9d'] * len(doc_vectors)

for i, (qv, cq) in enumerate(zip(query_vectors, colors_q)):
    ax2.arrow(0, 0, qv[0], qv[1], head_width=0.05, head_length=0.05, 
              fc=cq, ec=cq, linewidth=2, alpha=0.7, label='Query tokens' if i == 0 else '')

for i, (dv, cd) in enumerate(zip(doc_vectors, colors_d)):
    ax2.arrow(0, 0, dv[0], dv[1], head_width=0.05, head_length=0.05, 
              fc=cd, ec=cd, linewidth=1.5, alpha=0.5, label='Doc tokens' if i == 0 else '')

ax2.set_xlim(-0.1, 1.0)
ax2.set_ylim(-0.1, 1.0)
ax2.set_xlabel('Dimension 1', fontweight='bold')
ax2.set_ylabel('Dimension 2', fontweight='bold')
ax2.set_title(f'MaxSim (Late Interaction)\nOne vector per token\nMaxSim: {maxsim_total:.2f}', 
              fontweight='bold')
ax2.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig(output_dir / 'maxsim_vs_dense.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: maxsim_vs_dense.png")

print("\n✅ All MaxSim visualizations generated!")

