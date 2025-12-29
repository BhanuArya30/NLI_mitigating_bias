import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from helpers import compute_accuracy
import json

# Load SNLI dataset
dataset = datasets.load_dataset('snli')
dataset = dataset.filter(lambda ex: ex['label'] != -1)

# Modify dataset to use only hypothesis (set premise to empty or repeated hypothesis)
def hypothesis_only(examples):
    # Option 1: Empty premise
    examples['premise'] = [''] * len(examples['premise'])
    # Option 2: Repeat hypothesis as premise (uncomment to try)
    # examples['premise'] = examples['hypothesis']
    return examples

train_dataset = dataset['train'].map(hypothesis_only, batched=True)
eval_dataset = dataset['validation'].map(hypothesis_only, batched=True)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')

def tokenize(examples):
    return tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=128,
        padding='max_length'
    )

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['premise', 'hypothesis'])
eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=['premise', 'hypothesis'])

# Train hypothesis-only model
model = AutoModelForSequenceClassification.from_pretrained(
    'google/electra-small-discriminator',
    num_labels=3
)

training_args = TrainingArguments(
    output_dir='./hypothesis_only_model/',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    logging_steps=500,
    eval_strategy='epoch',
    save_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_accuracy,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"Hypothesis-only accuracy: {results['eval_accuracy']:.4f}")

# Save results
with open('./hypothesis_only_results.json', 'w') as f:
    json.dump(results, f, indent=2)