import datasets
import json
from collections import Counter, defaultdict
import numpy as np

# Load SNLI
dataset = datasets.load_dataset('snli')
dataset = dataset.filter(lambda ex: ex['label'] != -1)

# Load your model's predictions
with open('./eval_output/eval_predictions.jsonl', 'r') as f:
    predictions = [json.loads(line) for line in f]

label_names = ['entailment', 'neutral', 'contradiction']

# Analyze unigrams and bigrams in hypotheses
def extract_ngrams(text, n=1):
    words = text.lower().split()
    if n == 1:
        return words
    else:
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

# Count ngrams per label
ngram_label_counts = defaultdict(lambda: defaultdict(int))
label_counts = defaultdict(int)

for ex in predictions:
    label = ex['label']
    label_counts[label] += 1
    
    # Get unigrams and bigrams
    for ngram in extract_ngrams(ex['hypothesis'], n=1):
        ngram_label_counts[ngram][label] += 1
    for ngram in extract_ngrams(ex['hypothesis'], n=2):
        ngram_label_counts[ngram][label] += 1

# Calculate PMI (Pointwise Mutual Information) for each ngram-label pair
# PMI(ngram, label) = log(P(ngram, label) / (P(ngram) * P(label)))
total = sum(label_counts.values())
ngram_pmis = defaultdict(lambda: {})

for ngram, label_dist in ngram_label_counts.items():
    ngram_total = sum(label_dist.values())
    if ngram_total < 50:  # Skip rare ngrams
        continue
    
    for label in range(3):
        if label in label_dist:
            p_ngram_label = label_dist[label] / total
            p_ngram = ngram_total / total
            p_label = label_counts[label] / total
            
            pmi = np.log(p_ngram_label / (p_ngram * p_label))
            ngram_pmis[label][ngram] = pmi

# Print top correlations for each label
print("\n=== Top N-grams Correlated with Each Label ===\n")
for label in range(3):
    print(f"\n{label_names[label].upper()}:")
    sorted_ngrams = sorted(ngram_pmis[label].items(), key=lambda x: x[1], reverse=True)[:15]
    for ngram, pmi in sorted_ngrams:
        count = ngram_label_counts[ngram][label]
        print(f"  '{ngram}': PMI={pmi:.3f}, count={count}")

# Save to file
with open('./ngram_analysis.json', 'w') as f:
    json.dump({
        'label_counts': dict(label_counts),
        'top_correlations': {
            label_names[label]: [
                {'ngram': ng, 'pmi': float(pmi), 'count': ngram_label_counts[ng][label]}
                for ng, pmi in sorted(ngram_pmis[label].items(), key=lambda x: x[1], reverse=True)[:20]
            ]
            for label in range(3)
        }
    }, f, indent=2)

print("\n✓ Analysis saved to ngram_analysis.json")