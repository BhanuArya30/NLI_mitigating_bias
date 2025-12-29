import json
from collections import defaultdict
import random

# Load predictions
with open('./eval_output/eval_predictions.jsonl', 'r') as f:
    predictions = [json.loads(line) for line in f]

label_names = ['entailment', 'neutral', 'contradiction']

# Find errors
errors = [p for p in predictions if p['predicted_label'] != p['label']]
correct = [p for p in predictions if p['predicted_label'] == p['label']]

print(f"Total examples: {len(predictions)}")
print(f"Correct: {len(correct)} ({100*len(correct)/len(predictions):.2f}%)")
print(f"Errors: {len(errors)} ({100*len(errors)/len(predictions):.2f}%)")

# Analyze error patterns
error_patterns = defaultdict(list)

for err in errors:
    true_label = label_names[err['label']]
    pred_label = label_names[err['predicted_label']]
    pattern = f"{true_label} → {pred_label}"
    error_patterns[pattern].append(err)

print("\n=== Error Confusion Matrix ===")
for pattern, examples in sorted(error_patterns.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"{pattern}: {len(examples)} errors")

# Look for specific patterns
print("\n=== Examples with Negation Words (not, no, never, nobody) ===")
negation_words = ['not', 'no', 'never', 'nobody', 'nothing', "n't"]

negation_errors = []
for err in errors:
    hyp_lower = err['hypothesis'].lower()
    if any(word in hyp_lower for word in negation_words):
        negation_errors.append(err)

print(f"Errors with negation: {len(negation_errors)}/{len(errors)}")

# Show some examples
print("\nSample negation errors:")
for ex in random.sample(negation_errors, min(5, len(negation_errors))):
    print(f"\nPremise: {ex['premise']}")
    print(f"Hypothesis: {ex['hypothesis']}")
    print(f"True: {label_names[ex['label']]} | Predicted: {label_names[ex['predicted_label']]}")

# Analyze hypothesis length
print("\n=== Length Analysis ===")
error_lengths = [len(ex['hypothesis'].split()) for ex in errors]
correct_lengths = [len(ex['hypothesis'].split()) for ex in correct]

print(f"Avg error hypothesis length: {sum(error_lengths)/len(error_lengths):.2f}")
print(f"Avg correct hypothesis length: {sum(correct_lengths)/len(correct_lengths):.2f}")

# Find examples with high overlap
print("\n=== Lexical Overlap Analysis ===")
def word_overlap(premise, hypothesis):
    p_words = set(premise.lower().split())
    h_words = set(hypothesis.lower().split())
    if len(h_words) == 0:
        return 0
    return len(p_words & h_words) / len(h_words)

high_overlap_errors = [ex for ex in errors if word_overlap(ex['premise'], ex['hypothesis']) > 0.7]
low_overlap_errors = [ex for ex in errors if word_overlap(ex['premise'], ex['hypothesis']) < 0.3]

print(f"High overlap (>70%) errors: {len(high_overlap_errors)}")
print(f"Low overlap (<30%) errors: {len(low_overlap_errors)}")

# Save detailed error analysis
with open('./error_analysis.json', 'w') as f:
    json.dump({
        'summary': {
            'total': len(predictions),
            'correct': len(correct),
            'errors': len(errors),
            'accuracy': len(correct) / len(predictions)
        },
        'error_patterns': {k: len(v) for k, v in error_patterns.items()},
        'negation_errors': len(negation_errors),
        'sample_errors': random.sample(errors, min(20, len(errors)))
    }, f, indent=2)

print("\n✓ Detailed analysis saved to error_analysis.json")