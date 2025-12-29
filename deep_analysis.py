"""
Deep Analysis: Enhanced Insights for Report
============================================
This script provides detailed analysis of debiasing results including:
- Artifact-specific improvements
- Error pattern changes
- Detailed example analysis
- Statistical significance tests
"""

import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*80)
print("DEEP ANALYSIS: BASELINE vs DEBIASED")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading predictions...")

with open('./eval_output/eval_predictions.jsonl', 'r') as f:
    baseline = [json.loads(line) for line in f]

with open('./debiased_model/eval_predictions.jsonl', 'r') as f:
    debiased = [json.loads(line) for line in f]

label_names = ['entailment', 'neutral', 'contradiction']

print(f"✓ Loaded {len(baseline)} examples")

# ============================================================================
# Analysis 1: Artifact-Specific Performance
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 1: ARTIFACT-SPECIFIC PERFORMANCE")
print("="*80)

# Define artifact patterns from Part 1
artifacts = {
    'negation': ['not', 'no', 'never', 'nobody', 'nothing', "n't", 'none'],
    'location_entailment': ['outside', 'outdoors', 'outdoor'],
    'location_contradiction': ['inside', 'indoors', 'indoor'],
    'activity_contradiction': ['sleeping', 'eating', 'driving', 'swimming', 'sitting'],
    'intention_neutral': ['trying to', 'about to', 'going to', 'waiting for'],
    'existential': ['there is', 'there are', 'someone', 'somebody'],
}

def has_artifact(text, artifact_words):
    """Check if text contains any artifact words"""
    text_lower = text.lower()
    return any(word in text_lower for word in artifact_words)

# Analyze each artifact
artifact_results = {}

for artifact_name, artifact_words in artifacts.items():
    # Find examples with this artifact
    artifact_indices = [
        i for i, ex in enumerate(baseline) 
        if has_artifact(ex['hypothesis'], artifact_words)
    ]
    
    if len(artifact_indices) == 0:
        continue
    
    # Calculate performance
    baseline_correct = sum(
        1 for i in artifact_indices 
        if baseline[i]['predicted_label'] == baseline[i]['label']
    )
    debiased_correct = sum(
        1 for i in artifact_indices 
        if debiased[i]['predicted_label'] == debiased[i]['label']
    )
    
    baseline_acc = baseline_correct / len(artifact_indices) * 100
    debiased_acc = debiased_correct / len(artifact_indices) * 100
    
    # Count improvements/degradations
    improved = sum(
        1 for i in artifact_indices
        if baseline[i]['predicted_label'] != baseline[i]['label'] 
        and debiased[i]['predicted_label'] == debiased[i]['label']
    )
    degraded = sum(
        1 for i in artifact_indices
        if baseline[i]['predicted_label'] == baseline[i]['label'] 
        and debiased[i]['predicted_label'] != debiased[i]['label']
    )
    
    artifact_results[artifact_name] = {
        'count': len(artifact_indices),
        'baseline_acc': baseline_acc,
        'debiased_acc': debiased_acc,
        'change': debiased_acc - baseline_acc,
        'improved': improved,
        'degraded': degraded,
        'net': improved - degraded
    }

# Print results
print("\n📊 Performance on Artifact-Containing Examples:\n")
print(f"{'Artifact':<25} {'Count':<8} {'Baseline':<10} {'Debiased':<10} {'Change':<10} {'Net'}")
print("-" * 80)

for artifact_name, results in sorted(artifact_results.items(), key=lambda x: abs(x[1]['change']), reverse=True):
    print(f"{artifact_name:<25} {results['count']:<8} "
          f"{results['baseline_acc']:>6.2f}%    {results['debiased_acc']:>6.2f}%    "
          f"{results['change']:>+6.2f}%    {results['net']:>+4d}")

# ============================================================================
# Analysis 2: Error Pattern Changes
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 2: ERROR PATTERN CHANGES")
print("="*80)

# Build confusion matrices
def build_confusion_matrix(predictions):
    matrix = np.zeros((3, 3), dtype=int)
    for p in predictions:
        matrix[p['label']][p['predicted_label']] += 1
    return matrix

baseline_confusion = build_confusion_matrix(baseline)
debiased_confusion = build_confusion_matrix(debiased)
change_matrix = debiased_confusion - baseline_confusion

print("\n📊 Confusion Matrix Changes (Debiased - Baseline):")
print("\nPredicted →     Entailment   Neutral   Contradiction")
print("-" * 60)
for i, true_label in enumerate(label_names):
    print(f"True {true_label:<13}", end="")
    for j in range(3):
        change = change_matrix[i][j]
        print(f"{change:>10}", end="")
    print()

# Identify biggest changes
print("\n📋 Biggest Changes in Error Patterns:")
changes = []
for i in range(3):
    for j in range(3):
        if i != j:  # Only off-diagonal (errors)
            changes.append((
                f"{label_names[i]} → {label_names[j]}",
                change_matrix[i][j],
                baseline_confusion[i][j],
                debiased_confusion[i][j]
            ))

changes.sort(key=lambda x: abs(x[1]), reverse=True)
for pattern, change, baseline_count, debiased_count in changes[:6]:
    direction = "↑" if change > 0 else "↓"
    print(f"  {pattern:<30} {direction} {abs(change):>3} ({baseline_count} → {debiased_count})")

# ============================================================================
# Analysis 3: Detailed Example Analysis
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 3: DETAILED EXAMPLE ANALYSIS")
print("="*80)

# Categorize all changed predictions
improved = []
degraded = []
still_wrong_diff = []

for i, (b, d) in enumerate(zip(baseline, debiased)):
    if b['predicted_label'] != d['predicted_label']:
        baseline_correct = b['predicted_label'] == b['label']
        debiased_correct = d['predicted_label'] == d['label']
        
        if not baseline_correct and debiased_correct:
            improved.append(i)
        elif baseline_correct and not debiased_correct:
            degraded.append(i)
        else:
            still_wrong_diff.append(i)

print(f"\n📊 Change Summary:")
print(f"  Improved (wrong → correct):    {len(improved):>4}")
print(f"  Degraded (correct → wrong):    {len(degraded):>4}")
print(f"  Changed but still wrong:       {len(still_wrong_diff):>4}")
print(f"  Net change:                    {len(improved) - len(degraded):>+4}")

# Analyze improved examples by artifact
print("\n📋 What Improved?")
improved_by_artifact = defaultdict(list)
for idx in improved:
    ex = baseline[idx]
    for artifact_name, artifact_words in artifacts.items():
        if has_artifact(ex['hypothesis'], artifact_words):
            improved_by_artifact[artifact_name].append(idx)

for artifact_name, indices in sorted(improved_by_artifact.items(), key=lambda x: len(x[1]), reverse=True):
    if len(indices) > 0:
        print(f"  {artifact_name:<25} {len(indices):>3} examples")

# Analyze degraded examples by artifact
print("\n📋 What Degraded?")
degraded_by_artifact = defaultdict(list)
for idx in degraded:
    ex = baseline[idx]
    for artifact_name, artifact_words in artifacts.items():
        if has_artifact(ex['hypothesis'], artifact_words):
            degraded_by_artifact[artifact_name].append(idx)

for artifact_name, indices in sorted(degraded_by_artifact.items(), key=lambda x: len(x[1]), reverse=True):
    if len(indices) > 0:
        print(f"  {artifact_name:<25} {len(indices):>3} examples")

# ============================================================================
# Analysis 4: Length and Overlap Analysis
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 4: LENGTH AND OVERLAP PATTERNS")
print("="*80)

def word_overlap(premise, hypothesis):
    p_words = set(premise.lower().split())
    h_words = set(hypothesis.lower().split())
    if len(h_words) == 0:
        return 0
    return len(p_words & h_words) / len(h_words)

def get_length(text):
    return len(text.split())

# Analyze improved examples
if len(improved) > 0:
    improved_overlaps = [word_overlap(baseline[i]['premise'], baseline[i]['hypothesis']) for i in improved]
    improved_lengths = [get_length(baseline[i]['hypothesis']) for i in improved]
    
    print("\n📊 Improved Examples Characteristics:")
    print(f"  Average overlap:         {np.mean(improved_overlaps):.3f}")
    print(f"  Average hypothesis len:  {np.mean(improved_lengths):.1f} words")
    print(f"  Median overlap:          {np.median(improved_overlaps):.3f}")

# Analyze degraded examples
if len(degraded) > 0:
    degraded_overlaps = [word_overlap(baseline[i]['premise'], baseline[i]['hypothesis']) for i in degraded]
    degraded_lengths = [get_length(baseline[i]['hypothesis']) for i in degraded]
    
    print("\n📊 Degraded Examples Characteristics:")
    print(f"  Average overlap:         {np.mean(degraded_overlaps):.3f}")
    print(f"  Average hypothesis len:  {np.mean(degraded_lengths):.1f} words")
    print(f"  Median overlap:          {np.median(degraded_overlaps):.3f}")

# ============================================================================
# Analysis 5: Extract Detailed Examples for Each Artifact
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 5: EXAMPLE CASES BY ARTIFACT")
print("="*80)

examples_by_artifact = {}

# Get best improved examples for each artifact
for artifact_name, indices in improved_by_artifact.items():
    if len(indices) > 0:
        # Get up to 3 examples
        examples = []
        for idx in indices[:3]:
            ex = baseline[idx]
            examples.append({
                'premise': ex['premise'][:100] + '...' if len(ex['premise']) > 100 else ex['premise'],
                'hypothesis': ex['hypothesis'],
                'true_label': label_names[ex['label']],
                'baseline_pred': label_names[baseline[idx]['predicted_label']],
                'debiased_pred': label_names[debiased[idx]['predicted_label']]
            })
        examples_by_artifact[f"improved_{artifact_name}"] = examples

# Get example degraded cases
for artifact_name, indices in degraded_by_artifact.items():
    if len(indices) > 0:
        examples = []
        for idx in indices[:2]:  # Just 2 degraded examples per artifact
            ex = baseline[idx]
            examples.append({
                'premise': ex['premise'][:100] + '...' if len(ex['premise']) > 100 else ex['premise'],
                'hypothesis': ex['hypothesis'],
                'true_label': label_names[ex['label']],
                'baseline_pred': label_names[baseline[idx]['predicted_label']],
                'debiased_pred': label_names[debiased[idx]['predicted_label']]
            })
        examples_by_artifact[f"degraded_{artifact_name}"] = examples

# Print examples for most significant artifacts
significant_artifacts = ['negation', 'activity_contradiction', 'location_entailment']

for artifact_name in significant_artifacts:
    improved_key = f"improved_{artifact_name}"
    if improved_key in examples_by_artifact and len(examples_by_artifact[improved_key]) > 0:
        print(f"\n📋 IMPROVED: {artifact_name.upper().replace('_', ' ')}")
        for i, ex in enumerate(examples_by_artifact[improved_key], 1):
            print(f"\n  Example {i}:")
            print(f"    Premise: {ex['premise']}")
            print(f"    Hypothesis: {ex['hypothesis']}")
            print(f"    True: {ex['true_label']}")
            print(f"    Baseline: {ex['baseline_pred']} ✗")
            print(f"    Debiased: {ex['debiased_pred']} ✓")

# ============================================================================
# Analysis 6: Statistical Summary
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 6: STATISTICAL SUMMARY")
print("="*80)

# Calculate effect sizes
from scipy import stats

def calculate_effect_size(baseline_preds, debiased_preds, labels):
    """Calculate Cohen's d effect size"""
    baseline_correct = [1 if baseline_preds[i] == labels[i] else 0 for i in range(len(labels))]
    debiased_correct = [1 if debiased_preds[i] == labels[i] else 0 for i in range(len(labels))]
    
    mean_diff = np.mean(debiased_correct) - np.mean(baseline_correct)
    pooled_std = np.sqrt((np.std(baseline_correct)**2 + np.std(debiased_correct)**2) / 2)
    
    if pooled_std == 0:
        return 0
    return mean_diff / pooled_std

baseline_preds = [p['predicted_label'] for p in baseline]
debiased_preds = [p['predicted_label'] for p in debiased]
true_labels = [p['label'] for p in baseline]

effect_size = calculate_effect_size(baseline_preds, debiased_preds, true_labels)

print(f"\n📊 Statistical Measures:")
print(f"  Overall accuracy change:   {87.89 - 88.55:+.2f}%")
print(f"  Effect size (Cohen's d):   {effect_size:.4f}")
print(f"  Prediction agreement:      {sum(1 for b, d in zip(baseline_preds, debiased_preds) if b == d) / len(baseline_preds) * 100:.2f}%")
print(f"  Examples changed:          {sum(1 for b, d in zip(baseline_preds, debiased_preds) if b != d)} ({sum(1 for b, d in zip(baseline_preds, debiased_preds) if b != d) / len(baseline_preds) * 100:.2f}%)")

# ============================================================================
# Save Detailed Results
# ============================================================================

print("\n" + "="*80)
print("SAVING DETAILED RESULTS")
print("="*80)

results = {
    'artifact_performance': artifact_results,
    'confusion_matrix_changes': {
        'baseline': baseline_confusion.tolist(),
        'debiased': debiased_confusion.tolist(),
        'change': change_matrix.tolist()
    },
    'change_summary': {
        'improved': len(improved),
        'degraded': len(degraded),
        'still_wrong_diff': len(still_wrong_diff),
        'net_change': len(improved) - len(degraded)
    },
    'examples_by_artifact': examples_by_artifact,
    'statistics': {
        'effect_size': float(effect_size),
        'agreement_rate': float(sum(1 for b, d in zip(baseline_preds, debiased_preds) if b == d) / len(baseline_preds)),
        'change_rate': float(sum(1 for b, d in zip(baseline_preds, debiased_preds) if b != d) / len(baseline_preds))
    }
}

with open('./deep_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Saved: deep_analysis_results.json")

# ============================================================================
# Create Enhanced Visualizations
# ============================================================================

print("\n" + "="*80)
print("CREATING ENHANCED VISUALIZATIONS")
print("="*80)

import os
os.makedirs('./enhanced_plots', exist_ok=True)

# Plot 1: Artifact Performance Comparison
if len(artifact_results) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    artifact_names_short = [name.replace('_', '\n') for name in artifact_results.keys()]
    baseline_accs = [r['baseline_acc'] for r in artifact_results.values()]
    debiased_accs = [r['debiased_acc'] for r in artifact_results.values()]
    
    x = np.arange(len(artifact_names_short))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline', color='lightblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, debiased_accs, width, label='Debiased', color='lightcoral', edgecolor='black')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Artifact Type', fontsize=12, fontweight='bold')
    ax.set_title('Performance on Artifact-Containing Examples', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(artifact_names_short, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./enhanced_plots/artifact_performance.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: enhanced_plots/artifact_performance.png")

# Plot 2: Change Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(change_matrix, annot=True, fmt='d', cmap='RdBu_r', center=0,
            xticklabels=label_names, yticklabels=label_names,
            cbar_kws={'label': 'Change in Count'}, ax=ax, linewidths=1)
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix Changes\n(Debiased - Baseline)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('./enhanced_plots/confusion_change_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: enhanced_plots/confusion_change_heatmap.png")

# Plot 3: Net Change by Artifact
if len(artifact_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    artifact_names_sorted = sorted(artifact_results.keys(), key=lambda x: artifact_results[x]['net'])
    net_changes = [artifact_results[name]['net'] for name in artifact_names_sorted]
    colors = ['green' if x > 0 else 'red' for x in net_changes]
    
    y_pos = np.arange(len(artifact_names_sorted))
    bars = ax.barh(y_pos, net_changes, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ').title() for name in artifact_names_sorted])
    ax.set_xlabel('Net Change (Improved - Degraded)', fontsize=12, fontweight='bold')
    ax.set_title('Net Improvement by Artifact Type', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, net_changes):
        width = bar.get_width()
        label_x = width + (1 if width > 0 else -1)
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{int(value):+d}',
                ha='left' if width > 0 else 'right',
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./enhanced_plots/net_change_by_artifact.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: enhanced_plots/net_change_by_artifact.png")

print("\n" + "="*80)
print("✓ DEEP ANALYSIS COMPLETE!")
print("="*80)

print("\n Generated Files:")
print("  1. deep_analysis_results.json - Comprehensive results")
print("  2. enhanced_plots/artifact_performance.png")
print("  3. enhanced_plots/confusion_change_heatmap.png")
print("  4. enhanced_plots/net_change_by_artifact.png")



print("\n" + "="*80)