import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create output directory for plots
import os
os.makedirs('./plots', exist_ok=True)

print("Loading data...")

# Load predictions
with open('./eval_output/eval_predictions.jsonl', 'r') as f:
    predictions = [json.loads(line) for line in f]

# Load ngram analysis
with open('./ngram_analysis.json', 'r') as f:
    ngram_data = json.load(f)

label_names = ['entailment', 'neutral', 'contradiction']

# Separate correct and incorrect predictions
errors = [p for p in predictions if p['predicted_label'] != p['label']]
correct = [p for p in predictions if p['predicted_label'] == p['label']]

print(f"Total: {len(predictions)}, Correct: {len(correct)}, Errors: {len(errors)}")

# ============================================================================
# VISUALIZATION 1: Model Performance Comparison
# ============================================================================
print("\n1. Creating performance comparison bar chart...")

models = ['Random\nBaseline', 'Hypothesis\nOnly', 'Full Model\n(Ours)']
accuracies = [33.3, 70.39, 88.55]
colors = ['#d62728', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('SNLI Model Performance Comparison\nHypothesis-Only Achieves 70% Without Premise!', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=33.3, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./plots/1_performance_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved: plots/1_performance_comparison.png")

# ============================================================================
# VISUALIZATION 2: Confusion Matrix
# ============================================================================
print("\n2. Creating confusion matrix...")

# Build confusion matrix
conf_matrix = np.zeros((3, 3), dtype=int)
for p in predictions:
    conf_matrix[p['label']][p['predicted_label']] += 1

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_names, yticklabels=label_names,
            cbar_kws={'label': 'Count'}, ax=ax, linewidths=1, linecolor='black')

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix: Full Model on SNLI Validation', fontsize=14, fontweight='bold')

# Add accuracy percentages
for i in range(3):
    total = conf_matrix[i].sum()
    acc = conf_matrix[i][i] / total * 100
    ax.text(i + 0.5, i - 0.3, f'{acc:.1f}%', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig('./plots/2_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   Saved: plots/2_confusion_matrix.png")

# ============================================================================
# VISUALIZATION 3: Error Distribution
# ============================================================================
print("\n3. Creating error distribution chart...")

error_types = defaultdict(int)
for err in errors:
    true_label = label_names[err['label']]
    pred_label = label_names[err['predicted_label']]
    pattern = f"{true_label} → {pred_label}"
    error_types[pattern] += 1

# Sort by count
sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
patterns, counts = zip(*sorted_errors)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(patterns, counts, color='coral', edgecolor='black', linewidth=1)

# Add value labels
for bar, count in zip(bars, counts):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f' {count} ({count/len(errors)*100:.1f}%)',
            ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Number of Errors', fontsize=12, fontweight='bold')
ax.set_title('Error Distribution by Confusion Type\nNeutral Class is Most Confused', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('./plots/3_error_distribution.png', dpi=300, bbox_inches='tight')
print("   Saved: plots/3_error_distribution.png")

# ============================================================================
# VISUALIZATION 4: Top N-gram Artifacts
# ============================================================================
print("\n4. Creating n-gram artifacts chart...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for idx, label in enumerate(['entailment', 'neutral', 'contradiction']):
    top_ngrams = ngram_data['top_correlations'][label][:10]  # Top 10
    ngrams = [item['ngram'] for item in top_ngrams]
    pmis = [item['pmi'] for item in top_ngrams]
    
    # Color code by PMI strength
    colors_pmi = plt.cm.RdYlGn(np.array(pmis) / max(pmis))
    
    axes[idx].barh(ngrams, pmis, color=colors_pmi, edgecolor='black', linewidth=1)
    axes[idx].set_xlabel('PMI Score', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{label.upper()}\nTop Artifacts', fontsize=12, fontweight='bold')
    axes[idx].invert_yaxis()
    axes[idx].grid(axis='x', alpha=0.3)
    
    # Add PMI values
    for i, (ngram, pmi) in enumerate(zip(ngrams, pmis)):
        axes[idx].text(pmi, i, f' {pmi:.2f}', 
                      va='center', fontsize=9, fontweight='bold')

plt.suptitle('Top N-gram Artifacts by Label (PMI Score)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./plots/4_ngram_artifacts.png', dpi=300, bbox_inches='tight')
print("   Saved: plots/4_ngram_artifacts.png")

# ============================================================================
# VISUALIZATION 5: Lexical Overlap Analysis
# ============================================================================
print("\n5. Creating lexical overlap analysis...")

def word_overlap(premise, hypothesis):
    p_words = set(premise.lower().split())
    h_words = set(hypothesis.lower().split())
    if len(h_words) == 0:
        return 0
    return len(p_words & h_words) / len(h_words)

# Calculate overlap for all examples
overlaps_correct = [word_overlap(p['premise'], p['hypothesis']) for p in correct]
overlaps_errors = [word_overlap(p['premise'], p['hypothesis']) for p in errors]

# Create bins
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

correct_hist, _ = np.histogram(overlaps_correct, bins=bins)
errors_hist, _ = np.histogram(overlaps_errors, bins=bins)

# Normalize to get error rate per bin
total_hist = correct_hist + errors_hist
error_rate = errors_hist / total_hist * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Distribution
x = np.arange(len(bin_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, correct_hist, width, label='Correct', 
               color='lightgreen', edgecolor='black', linewidth=1)
bars2 = ax1.bar(x + width/2, errors_hist, width, label='Errors', 
               color='lightcoral', edgecolor='black', linewidth=1)

ax1.set_xlabel('Lexical Overlap', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Distribution by Lexical Overlap', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(bin_labels)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Error rate
bars = ax2.bar(bin_labels, error_rate, color='orangered', 
              edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.axhline(y=11.45, color='blue', linestyle='--', linewidth=2, 
           label='Overall error rate (11.45%)')
ax2.set_xlabel('Lexical Overlap', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Error Rate by Lexical Overlap\nLow Overlap = More Errors', 
             fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, rate in zip(bars, error_rate):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('./plots/5_lexical_overlap_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: plots/5_lexical_overlap_analysis.png")

# ============================================================================
# VISUALIZATION 6: Hypothesis Length Analysis
# ============================================================================
print("\n6. Creating hypothesis length analysis...")

lengths_correct = [len(p['hypothesis'].split()) for p in correct]
lengths_errors = [len(p['hypothesis'].split()) for p in errors]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Distributions
axes[0].hist(lengths_correct, bins=20, alpha=0.6, label='Correct', 
            color='green', edgecolor='black', density=True)
axes[0].hist(lengths_errors, bins=20, alpha=0.6, label='Errors', 
            color='red', edgecolor='black', density=True)
axes[0].axvline(np.mean(lengths_correct), color='green', linestyle='--', 
               linewidth=2, label=f'Mean Correct: {np.mean(lengths_correct):.2f}')
axes[0].axvline(np.mean(lengths_errors), color='red', linestyle='--', 
               linewidth=2, label=f'Mean Error: {np.mean(lengths_errors):.2f}')
axes[0].set_xlabel('Hypothesis Length (words)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
axes[0].set_title('Hypothesis Length Distribution', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Plot 2: Error rate by length bin
length_bins = range(0, 20, 2)
bin_labels_len = [f'{i}-{i+2}' for i in length_bins[:-1]]

correct_by_len = defaultdict(int)
errors_by_len = defaultdict(int)

for p in correct:
    length = len(p['hypothesis'].split())
    bin_idx = min(length // 2, len(length_bins) - 2)
    correct_by_len[bin_idx] += 1

for p in errors:
    length = len(p['hypothesis'].split())
    bin_idx = min(length // 2, len(length_bins) - 2)
    errors_by_len[bin_idx] += 1

error_rates_by_len = []
for i in range(len(bin_labels_len)):
    total = correct_by_len[i] + errors_by_len[i]
    if total > 0:
        error_rates_by_len.append(errors_by_len[i] / total * 100)
    else:
        error_rates_by_len.append(0)

bars = axes[1].bar(bin_labels_len, error_rates_by_len, color='purple', 
                  edgecolor='black', linewidth=1, alpha=0.7)
axes[1].axhline(y=11.45, color='blue', linestyle='--', linewidth=2, 
               label='Overall error rate')
axes[1].set_xlabel('Hypothesis Length (words)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Error Rate by Hypothesis Length', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('./plots/6_length_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: plots/6_length_analysis.png")

# ============================================================================
# VISUALIZATION 7: Label Distribution and Per-Class Accuracy
# ============================================================================
print("\n7. Creating label distribution and accuracy by class...")

# Count labels
label_counts = Counter([p['label'] for p in predictions])
label_correct = Counter([p['label'] for p in correct])

# Calculate per-class accuracy
per_class_acc = {label: (label_correct[label] / label_counts[label] * 100) 
                 for label in range(3)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Label distribution
colors_dist = ['#2ca02c', '#ff7f0e', '#d62728']
bars1 = ax1.bar(label_names, [label_counts[i] for i in range(3)], 
               color=colors_dist, edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Label Distribution in SNLI Validation Set', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add counts
for bar, label_id in zip(bars1, range(3)):
    height = bar.get_height()
    count = label_counts[label_id]
    pct = count / len(predictions) * 100
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Per-class accuracy
bars2 = ax2.bar(label_names, [per_class_acc[i] for i in range(3)], 
               color=colors_dist, edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.axhline(y=88.55, color='blue', linestyle='--', linewidth=2, 
           label='Overall accuracy (88.55%)')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
ax2.set_ylim(75, 95)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Add accuracy values
for bar, label_id in zip(bars2, range(3)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{per_class_acc[label_id]:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('./plots/7_label_distribution_accuracy.png', dpi=300, bbox_inches='tight')
print("   Saved: plots/7_label_distribution_accuracy.png")

# ============================================================================
# Summary Statistics Table
# ============================================================================
print("\n8. Generating summary statistics...")

summary = f"""
ANALYSIS SUMMARY
================

Overall Performance:
- Total examples: {len(predictions)}
- Accuracy: {len(correct)/len(predictions)*100:.2f}%
- Errors: {len(errors)} ({len(errors)/len(predictions)*100:.2f}%)

Hypothesis-Only Baseline:
- Accuracy: 70.39%
- Above random: 37.09 percentage points
- Artifact severity: HIGH

Per-Class Performance:
- Entailment: {per_class_acc[0]:.2f}%
- Neutral: {per_class_acc[1]:.2f}%
- Contradiction: {per_class_acc[2]:.2f}%

Top Artifacts:
- Strongest: 'nobody' (PMI=1.099, contradiction)
- Location: 'outdoors'/'outside' → entailment, 'inside' → contradiction
- Activity: 'sleeping', 'eating', 'driving' → contradiction
- Intention: 'trying to', 'about to' → neutral

Error Patterns:
- Most confused: Neutral class (44.3% of errors)
- Low overlap errors: 395 (35.1% of all errors)
- Negation errors: 51 (4.5% of all errors)
- Avg error length: 7.92 words vs 7.46 for correct

Key Insights:
1. Model relies heavily on hypothesis-only patterns (70% accuracy)
2. Strong lexical artifacts present (especially negation words)
3. Struggles with low lexical overlap (requires true inference)
4. Neutral class boundary is most difficult
"""

with open('./plots/SUMMARY.txt', 'w') as f:
    f.write(summary)

print(summary)
print("\n" + "="*80)
print("✓ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files in ./plots/:")
print("  1. 1_performance_comparison.png")
print("  2. 2_confusion_matrix.png")
print("  3. 3_error_distribution.png")
print("  4. 4_ngram_artifacts.png")
print("  5. 5_lexical_overlap_analysis.png")
print("  6. 6_length_analysis.png")
print("  7. 7_label_distribution_accuracy.png")
print("  8. SUMMARY.txt")

