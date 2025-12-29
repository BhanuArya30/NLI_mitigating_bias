"""
Evaluation Script: Comparing Baseline vs Debiased Model
========================================================
This script performs comprehensive evaluation comparing:
1. Baseline model (trained on full SNLI)
2. Debiased model (trained with hypothesis-only debiasing)

Key comparisons:
- Overall accuracy
- Performance on low/high lexical overlap examples
- Performance on examples where hypothesis-only model succeeds/fails
- Error pattern changes
- Per-label performance
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "="*80)
print("COMPARATIVE EVALUATION: BASELINE vs DEBIASED MODEL")
print("="*80)

# ============================================================================
# Load Predictions
# ============================================================================

print("\nLoading predictions...")

# Baseline predictions
with open('./eval_output/eval_predictions.jsonl', 'r') as f:
    baseline_preds = [json.loads(line) for line in f]

# Debiased predictions
if not os.path.exists('./debiased_model/eval_predictions.jsonl'):
    print("\n❌ Debiased model predictions not found!")
    print("Please run: python3 train_debiased_model.py first")
    exit(1)

with open('./debiased_model/eval_predictions.jsonl', 'r') as f:
    debiased_preds = [json.loads(line) for line in f]

# Hypothesis-only predictions (for analysis)
hypothesis_only_exists = os.path.exists('./hypothesis_only_results.json')
if hypothesis_only_exists:
    # We'll need to generate hypothesis-only predictions on eval set
    # For now, we'll use the model accuracy
    pass

label_names = ['entailment', 'neutral', 'contradiction']

print(f"✓ Loaded {len(baseline_preds)} baseline predictions")
print(f"✓ Loaded {len(debiased_preds)} debiased predictions")

# ============================================================================
# Overall Performance Comparison
# ============================================================================

print("\n" + "-"*80)
print("OVERALL PERFORMANCE")
print("-"*80)

def calculate_accuracy(predictions):
    correct = sum(1 for p in predictions if p['predicted_label'] == p['label'])
    return correct / len(predictions) * 100

baseline_acc = calculate_accuracy(baseline_preds)
debiased_acc = calculate_accuracy(debiased_preds)

print(f"\nBaseline Model:  {baseline_acc:.2f}%")
print(f"Debiased Model:  {debiased_acc:.2f}%")
print(f"Difference:      {debiased_acc - baseline_acc:+.2f}%")

# ============================================================================
# Lexical Overlap Analysis
# ============================================================================

print("\n" + "-"*80)
print("LEXICAL OVERLAP ANALYSIS")
print("-"*80)

def word_overlap(premise, hypothesis):
    p_words = set(premise.lower().split())
    h_words = set(hypothesis.lower().split())
    if len(h_words) == 0:
        return 0
    return len(p_words & h_words) / len(h_words)

# Categorize examples by overlap
low_overlap_indices = []
high_overlap_indices = []

for i, pred in enumerate(baseline_preds):
    overlap = word_overlap(pred['premise'], pred['hypothesis'])
    if overlap < 0.3:
        low_overlap_indices.append(i)
    elif overlap > 0.7:
        high_overlap_indices.append(i)

print(f"\nLow overlap examples (<30%): {len(low_overlap_indices)}")
print(f"High overlap examples (>70%): {len(high_overlap_indices)}")

# Calculate accuracy on each subset
def subset_accuracy(predictions, indices):
    subset = [predictions[i] for i in indices]
    return calculate_accuracy(subset)

baseline_low = subset_accuracy(baseline_preds, low_overlap_indices)
debiased_low = subset_accuracy(debiased_preds, low_overlap_indices)

baseline_high = subset_accuracy(baseline_preds, high_overlap_indices)
debiased_high = subset_accuracy(debiased_preds, high_overlap_indices)

print(f"\nLow Overlap (<30%):")
print(f"  Baseline:  {baseline_low:.2f}%")
print(f"  Debiased:  {debiased_low:.2f}%")
print(f"  Change:    {debiased_low - baseline_low:+.2f}%")

print(f"\nHigh Overlap (>70%):")
print(f"  Baseline:  {baseline_high:.2f}%")
print(f"  Debiased:  {debiased_high:.2f}%")
print(f"  Change:    {debiased_high - baseline_high:+.2f}%")

# ============================================================================
# Per-Label Performance
# ============================================================================

print("\n" + "-"*80)
print("PER-LABEL PERFORMANCE")
print("-"*80)

def per_label_accuracy(predictions):
    label_correct = defaultdict(int)
    label_total = defaultdict(int)
    
    for p in predictions:
        label = p['label']
        label_total[label] += 1
        if p['predicted_label'] == label:
            label_correct[label] += 1
    
    return {label: (label_correct[label] / label_total[label] * 100) 
            for label in range(3)}

baseline_per_label = per_label_accuracy(baseline_preds)
debiased_per_label = per_label_accuracy(debiased_preds)

print(f"\n{'Label':<15} {'Baseline':<12} {'Debiased':<12} {'Change'}")
print("-" * 55)
for label in range(3):
    name = label_names[label]
    base = baseline_per_label[label]
    debi = debiased_per_label[label]
    change = debi - base
    print(f"{name:<15} {base:>6.2f}%      {debi:>6.2f}%      {change:>+6.2f}%")

# ============================================================================
# Examples that Changed
# ============================================================================

print("\n" + "-"*80)
print("PREDICTION CHANGES")
print("-"*80)

# Find examples where predictions changed
changed_indices = []
for i in range(len(baseline_preds)):
    if baseline_preds[i]['predicted_label'] != debiased_preds[i]['predicted_label']:
        changed_indices.append(i)

print(f"\nTotal examples with changed predictions: {len(changed_indices)} ({len(changed_indices)/len(baseline_preds)*100:.2f}%)")

# Categorize changes
improved = []  # baseline wrong -> debiased correct
degraded = []  # baseline correct -> debiased wrong
still_wrong = []  # baseline wrong -> debiased wrong (but different)

for i in changed_indices:
    baseline_correct = baseline_preds[i]['predicted_label'] == baseline_preds[i]['label']
    debiased_correct = debiased_preds[i]['predicted_label'] == debiased_preds[i]['label']
    
    if not baseline_correct and debiased_correct:
        improved.append(i)
    elif baseline_correct and not debiased_correct:
        degraded.append(i)
    else:
        still_wrong.append(i)

print(f"\nImproved (wrong → correct):           {len(improved)} examples")
print(f"Degraded (correct → wrong):           {len(degraded)} examples")
print(f"Still wrong (wrong → different wrong): {len(still_wrong)} examples")
print(f"\nNet improvement: {len(improved) - len(degraded)} examples")

# ============================================================================
# Analyze Improved Examples
# ============================================================================

print("\n" + "-"*80)
print("ANALYZING IMPROVED EXAMPLES")
print("-"*80)

if len(improved) > 0:
    # Calculate average overlap for improved examples
    improved_overlaps = [word_overlap(baseline_preds[i]['premise'], 
                                      baseline_preds[i]['hypothesis']) 
                        for i in improved]
    
    print(f"\nImproved examples characteristics:")
    print(f"  Count: {len(improved)}")
    print(f"  Avg lexical overlap: {np.mean(improved_overlaps):.3f}")
    
    # Show a few examples
    print(f"\n📋 Sample Improved Examples:")
    for idx in improved[:3]:
        ex = baseline_preds[idx]
        print(f"\n  Example {idx}:")
        print(f"    Premise: {ex['premise'][:100]}...")
        print(f"    Hypothesis: {ex['hypothesis']}")
        print(f"    True label: {label_names[ex['label']]}")
        print(f"    Baseline predicted: {label_names[baseline_preds[idx]['predicted_label']]}")
        print(f"    Debiased predicted: {label_names[debiased_preds[idx]['predicted_label']]} ✓")

# ============================================================================
# Analyze Degraded Examples
# ============================================================================

print("\n" + "-"*80)
print("ANALYZING DEGRADED EXAMPLES")
print("-"*80)

if len(degraded) > 0:
    degraded_overlaps = [word_overlap(baseline_preds[i]['premise'], 
                                      baseline_preds[i]['hypothesis']) 
                        for i in degraded]
    
    print(f"\nDegraded examples characteristics:")
    print(f"  Count: {len(degraded)}")
    print(f"  Avg lexical overlap: {np.mean(degraded_overlaps):.3f}")
    
    # Show a few examples
    print(f"\n📋 Sample Degraded Examples:")
    for idx in degraded[:3]:
        ex = baseline_preds[idx]
        print(f"\n  Example {idx}:")
        print(f"    Premise: {ex['premise'][:100]}...")
        print(f"    Hypothesis: {ex['hypothesis']}")
        print(f"    True label: {label_names[ex['label']]}")
        print(f"    Baseline predicted: {label_names[baseline_preds[idx]['predicted_label']]} ✓")
        print(f"    Debiased predicted: {label_names[debiased_preds[idx]['predicted_label']]} ✗")

# ============================================================================
# Create Comparison Visualizations
# ============================================================================

print("\n" + "-"*80)
print("GENERATING COMPARISON VISUALIZATIONS")
print("-"*80)

os.makedirs('./comparison_plots', exist_ok=True)

# Plot 1: Overall accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Baseline\nModel', 'Debiased\nModel']
accuracies = [baseline_acc, debiased_acc]
colors = ['#2ca02c' if debiased_acc >= baseline_acc else '#d62728', '#1f77b4']

bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(80, 92)
ax.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('./comparison_plots/1_overall_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: comparison_plots/1_overall_comparison.png")

# Plot 2: Per-label comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(label_names))
width = 0.35

baseline_accs = [baseline_per_label[i] for i in range(3)]
debiased_accs = [debiased_per_label[i] for i in range(3)]

bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline', 
              color='lightblue', edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, debiased_accs, width, label='Debiased', 
              color='lightcoral', edgecolor='black', linewidth=1)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Per-Label Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(label_names)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(75, 95)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('./comparison_plots/2_per_label_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: comparison_plots/2_per_label_comparison.png")

# Plot 3: Overlap-stratified performance
fig, ax = plt.subplots(figsize=(12, 6))
categories = ['Low Overlap\n(<30%)', 'High Overlap\n(>70%)']
baseline_overlap_accs = [baseline_low, baseline_high]
debiased_overlap_accs = [debiased_low, debiased_high]

x = np.arange(len(categories))
bars1 = ax.bar(x - width/2, baseline_overlap_accs, width, label='Baseline',
              color='lightgreen', edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, debiased_overlap_accs, width, label='Debiased',
              color='lightyellow', edgecolor='black', linewidth=1)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance by Lexical Overlap\nLow Overlap = More Challenging', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('./comparison_plots/3_overlap_stratified.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: comparison_plots/3_overlap_stratified.png")

# Plot 4: Prediction changes breakdown
fig, ax = plt.subplots(figsize=(10, 6))
change_types = ['Improved\n(wrong→correct)', 'Still Wrong\n(wrong→wrong)', 'Degraded\n(correct→wrong)']
change_counts = [len(improved), len(still_wrong), len(degraded)]
change_colors = ['green', 'orange', 'red']

bars = ax.bar(change_types, change_counts, color=change_colors, 
             edgecolor='black', linewidth=1.5, alpha=0.7)
ax.set_ylabel('Number of Examples', fontsize=12, fontweight='bold')
ax.set_title('Prediction Changes: Baseline → Debiased', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, change_counts):
    height = bar.get_height()
    pct = count / len(changed_indices) * 100 if len(changed_indices) > 0 else 0
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('./comparison_plots/4_prediction_changes.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: comparison_plots/4_prediction_changes.png")

# ============================================================================
# Save Summary Report
# ============================================================================

summary_report = f"""
COMPARATIVE EVALUATION SUMMARY
==============================

OVERALL PERFORMANCE
-------------------
Baseline Model:    {baseline_acc:.2f}%
Debiased Model:    {debiased_acc:.2f}%
Change:            {debiased_acc - baseline_acc:+.2f}%

PER-LABEL PERFORMANCE
---------------------
Entailment:
  Baseline: {baseline_per_label[0]:.2f}%
  Debiased: {debiased_per_label[0]:.2f}%
  Change:   {debiased_per_label[0] - baseline_per_label[0]:+.2f}%

Neutral:
  Baseline: {baseline_per_label[1]:.2f}%
  Debiased: {debiased_per_label[1]:.2f}%
  Change:   {debiased_per_label[1] - baseline_per_label[1]:+.2f}%

Contradiction:
  Baseline: {baseline_per_label[2]:.2f}%
  Debiased: {debiased_per_label[2]:.2f}%
  Change:   {debiased_per_label[2] - baseline_per_label[2]:+.2f}%

LEXICAL OVERLAP ANALYSIS
-------------------------
Low Overlap (<30%):
  Baseline: {baseline_low:.2f}%
  Debiased: {debiased_low:.2f}%
  Change:   {debiased_low - baseline_low:+.2f}%

High Overlap (>70%):
  Baseline: {baseline_high:.2f}%
  Debiased: {debiased_high:.2f}%
  Change:   {debiased_high - baseline_high:+.2f}%

PREDICTION CHANGES
------------------
Total changed:        {len(changed_indices)} ({len(changed_indices)/len(baseline_preds)*100:.2f}%)
Improved:             {len(improved)}
Degraded:             {len(degraded)}
Still wrong:          {len(still_wrong)}
Net improvement:      {len(improved) - len(degraded)}

KEY INSIGHTS
------------
{'✓' if debiased_acc > baseline_acc else '✗'} Overall accuracy {'improved' if debiased_acc > baseline_acc else 'decreased'}
{'✓' if debiased_low > baseline_low else '✗'} Low-overlap performance {'improved' if debiased_low > baseline_low else 'decreased'}
{'✓' if len(improved) > len(degraded) else '✗'} Net prediction improvement: {len(improved) - len(degraded)} examples
"""

with open('./comparison_plots/EVALUATION_SUMMARY.txt', 'w') as f:
    f.write(summary_report)

print("\n" + "="*80)
print(summary_report)
print("="*80)

print("\n✓ Evaluation complete!")
print(f"\nGenerated files in ./comparison_plots/:")
print("  1. 1_overall_comparison.png")
print("  2. 2_per_label_comparison.png")
print("  3. 3_overlap_stratified.png")
print("  4. 4_prediction_changes.png")
print("  5. EVALUATION_SUMMARY.txt")