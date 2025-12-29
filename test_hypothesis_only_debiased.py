"""
Test Hypothesis-Only Performance on Debiased Model
===================================================
Critical test: Does the debiased model still perform well when given only hypotheses?

Expected result: Hypothesis-only accuracy should DROP significantly compared to baseline.
- Baseline hypothesis-only: ~70%
- Debiased hypothesis-only: Should approach ~50-60% (closer to random 33%)

This proves the model is no longer relying on hypothesis artifacts.
"""

import torch
import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import numpy as np
from collections import Counter

print("\n" + "="*80)
print("HYPOTHESIS-ONLY TEST: DEBIASED MODEL")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# ============================================================================
# Load Models and Data
# ============================================================================

print("\nLoading models and data...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')

# Load debiased model
debiased_model = AutoModelForSequenceClassification.from_pretrained(
    './debiased_model/',
    num_labels=3
)
debiased_model.to(device)
debiased_model.eval()

# Load baseline model for comparison
baseline_model = AutoModelForSequenceClassification.from_pretrained(
    './trained_model/',
    num_labels=3
)
baseline_model.to(device)
baseline_model.eval()

print("✓ Models loaded")

# Load SNLI validation set
dataset = datasets.load_dataset('snli')
dataset = dataset.filter(lambda ex: ex['label'] != -1)
eval_dataset = dataset['validation']

print(f"✓ Loaded {len(eval_dataset)} validation examples")

# ============================================================================
# Create Hypothesis-Only Version
# ============================================================================

print("\n" + "-"*80)
print("Creating hypothesis-only test set...")
print("-"*80)

def hypothesis_only(examples):
    """Remove premise information"""
    examples['premise'] = [''] * len(examples['premise'])
    return examples

eval_hyp_only = eval_dataset.map(hypothesis_only, batched=True)

def tokenize(examples):
    return tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=128,
        padding='max_length'
    )

eval_tokenized = eval_hyp_only.map(
    tokenize,
    batched=True,
    remove_columns=['premise', 'hypothesis']
)

print("✓ Hypothesis-only test set created")

# ============================================================================
# Test Both Models on Hypothesis-Only Input
# ============================================================================

print("\n" + "-"*80)
print("Testing models on hypothesis-only input...")
print("-"*80)

def evaluate_model_hyp_only(model, dataset, tokenized_dataset):
    """Evaluate model on hypothesis-only data"""
    predictions = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(tokenized_dataset), batch_size):
            batch = tokenized_dataset[i:i+batch_size]
            
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_labels = logits.argmax(dim=-1).cpu().numpy()
            
            predictions.extend(pred_labels)
            
            if (i // batch_size) % 20 == 0:
                print(f"  Processed {i}/{len(tokenized_dataset)} examples...")
    
    # Calculate accuracy
    correct = 0
    for pred, ex in zip(predictions, dataset):
        if pred == ex['label']:
            correct += 1
    
    accuracy = correct / len(dataset) * 100
    
    # Per-label accuracy
    per_label_correct = Counter()
    per_label_total = Counter()
    
    for pred, ex in zip(predictions, dataset):
        label = ex['label']
        per_label_total[label] += 1
        if pred == label:
            per_label_correct[label] += 1
    
    per_label_acc = {label: (per_label_correct[label] / per_label_total[label] * 100)
                     for label in range(3)}
    
    return accuracy, per_label_acc, predictions

# Test baseline model
print("\nTesting BASELINE model (hypothesis-only):")
baseline_hyp_acc, baseline_hyp_per_label, baseline_hyp_preds = evaluate_model_hyp_only(
    baseline_model, eval_dataset, eval_tokenized
)

# Test debiased model
print("\nTesting DEBIASED model (hypothesis-only):")
debiased_hyp_acc, debiased_hyp_per_label, debiased_hyp_preds = evaluate_model_hyp_only(
    debiased_model, eval_dataset, eval_tokenized
)

# ============================================================================
# Compare Results
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS-ONLY RESULTS")
print("="*80)

label_names = ['entailment', 'neutral', 'contradiction']

print(f"\n{'Model':<25} {'Accuracy':<12} {'vs Random':<15}")
print("-" * 52)
print(f"{'Random Baseline':<25} {'33.33%':<12} {'—':<15}")
print(f"{'Baseline (hypothesis-only)':<25} {baseline_hyp_acc:>6.2f}%    {baseline_hyp_acc - 33.33:>+6.2f}%")
print(f"{'Debiased (hypothesis-only)':<25} {debiased_hyp_acc:>6.2f}%    {debiased_hyp_acc - 33.33:>+6.2f}%")

print(f"\n{'Change (debiased - baseline):':<40} {debiased_hyp_acc - baseline_hyp_acc:>+6.2f}%")

print("\n" + "-"*80)
print("PER-LABEL HYPOTHESIS-ONLY PERFORMANCE")
print("-"*80)

print(f"\n{'Label':<15} {'Baseline':<12} {'Debiased':<12} {'Change'}")
print("-" * 55)
for label in range(3):
    name = label_names[label]
    base = baseline_hyp_per_label[label]
    debi = debiased_hyp_per_label[label]
    change = debi - base
    print(f"{name:<15} {base:>6.2f}%      {debi:>6.2f}%      {change:>+6.2f}%")

# ============================================================================
# Analysis
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

reduction = baseline_hyp_acc - debiased_hyp_acc

print(f"\nHypothesis-only bias REDUCTION: {reduction:.2f} percentage points")

if reduction > 5:
    print("✓ SUCCESS: Significant reduction in hypothesis-only performance!")
    print("  The debiased model is less reliant on hypothesis artifacts.")
elif reduction > 0:
    print("~ PARTIAL: Modest reduction in hypothesis-only performance.")
    print("  Some debiasing occurred but artifacts remain.")
else:
    print("✗ CONCERN: Hypothesis-only performance increased or stayed same.")
    print("  Debiasing may not have been effective.")

# Check if still above random
if debiased_hyp_acc > 60:
    print(f"\n⚠️  WARNING: Debiased model still achieves {debiased_hyp_acc:.1f}% hypothesis-only.")
    print("   This is significantly above random (33.3%), suggesting artifacts remain.")
elif debiased_hyp_acc > 50:
    print(f"\n✓ GOOD: Debiased model achieves {debiased_hyp_acc:.1f}% hypothesis-only.")
    print("   This is closer to random, though some artifacts may remain.")
else:
    print(f"\n✓ EXCELLENT: Debiased model achieves {debiased_hyp_acc:.1f}% hypothesis-only.")
    print("   This approaches random baseline, indicating effective debiasing.")

# ============================================================================
# Agreement Analysis
# ============================================================================

print("\n" + "-"*80)
print("PREDICTION AGREEMENT ANALYSIS")
print("-"*80)

# How often do the models agree/disagree on hypothesis-only predictions?
agreement = sum(1 for b, d in zip(baseline_hyp_preds, debiased_hyp_preds) if b == d)
agreement_rate = agreement / len(baseline_hyp_preds) * 100

print(f"\nModels agree on {agreement}/{len(baseline_hyp_preds)} predictions ({agreement_rate:.1f}%)")
print(f"Models disagree on {len(baseline_hyp_preds) - agreement} predictions ({100-agreement_rate:.1f}%)")

# ============================================================================
# Save Results
# ============================================================================

results = {
    'random_baseline': 33.33,
    'baseline_hypothesis_only': baseline_hyp_acc,
    'debiased_hypothesis_only': debiased_hyp_acc,
    'reduction': reduction,
    'baseline_per_label': baseline_hyp_per_label,
    'debiased_per_label': debiased_hyp_per_label,
    'agreement_rate': agreement_rate
}

with open('./hypothesis_only_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to hypothesis_only_comparison.json")

# ============================================================================
# Visualization
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Overall comparison
models = ['Random\nBaseline', 'Baseline\n(Hyp-Only)', 'Debiased\n(Hyp-Only)']
accuracies = [33.33, baseline_hyp_acc, debiased_hyp_acc]
colors = ['gray', 'red', 'green' if reduction > 5 else 'orange']

bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', 
              linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Hypothesis-Only Performance Comparison\nLower is Better (Less Bias)', 
             fontsize=14, fontweight='bold')
ax1.set_ylim(0, 80)
ax1.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random')
ax1.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Per-label comparison
x = np.arange(len(label_names))
width = 0.35

baseline_accs = [baseline_hyp_per_label[i] for i in range(3)]
debiased_accs = [debiased_hyp_per_label[i] for i in range(3)]

bars1 = ax2.bar(x - width/2, baseline_accs, width, label='Baseline (Hyp-Only)',
               color='lightcoral', edgecolor='black', linewidth=1)
bars2 = ax2.bar(x + width/2, debiased_accs, width, label='Debiased (Hyp-Only)',
               color='lightgreen', edgecolor='black', linewidth=1)

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Per-Label Hypothesis-Only Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(label_names)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./hypothesis_only_comparison.png', dpi=300, bbox_inches='tight')

print("\n✓ Visualization saved to hypothesis_only_comparison.png")

print("\n" + "="*80)
print("✓ HYPOTHESIS-ONLY TEST COMPLETE!")
print("="*80)

if reduction > 5:
    print("  'The debiasing approach successfully reduced hypothesis-only")
    print(f"   performance from {baseline_hyp_acc:.1f}% to {debiased_hyp_acc:.1f}%, indicating")
    print("   the model now relies more on premise-hypothesis interaction.'")
else:
    print("  'While debiasing was attempted, hypothesis-only performance")
    print("   remained high, suggesting additional strategies may be needed.'")