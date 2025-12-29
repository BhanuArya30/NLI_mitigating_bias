"""
Ensemble Debiasing for SNLI
============================
This script implements ensemble-based debiasing using a hypothesis-only "bias expert".

Approach (based on He et al. 2019, Clark et al. 2019):
1. Load the trained hypothesis-only model (bias expert)
2. Get bias model predictions on training data
3. Train main model to predict residuals/focus on premise-hypothesis interaction
4. Use focal loss or reweighting to emphasize examples where bias model fails

Key idea: If hypothesis-only model can predict correctly, force main model 
to look beyond hypothesis patterns and use premise information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from helpers import compute_accuracy
import json
import numpy as np
from typing import Dict, Optional
import os

# ============================================================================
# Custom Loss Functions for Debiasing
# ============================================================================

class DebiasedTrainer(Trainer):
    """
    Custom Trainer that implements debiasing using ensemble prediction.
    
    This version uses a simpler approach: we load the bias model and use it
    during training to guide the main model away from bias-only predictions.
    """
    
    def __init__(self, *args, bias_model_path=None, debiasing_strength=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_model = None
        self.debiasing_strength = debiasing_strength
        
        # Load bias model if provided
        if bias_model_path:
            print(f"Loading bias model from {bias_model_path}...")
            from transformers import AutoModelForSequenceClassification
            self.bias_model = AutoModelForSequenceClassification.from_pretrained(
                bias_model_path,
                num_labels=3
            )
            self.bias_model.eval()
            # Move to same device as main model
            if torch.cuda.is_available():
                self.bias_model.cuda()
            print("✓ Bias model loaded for ensemble debiasing")
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss with ensemble debiasing.
        
        If bias model is available, we:
        1. Get bias model's predictions
        2. Identify where bias model is confident
        3. Push main model's predictions away from bias predictions
        """
        labels = inputs.get("labels")
        
        # Forward pass on main model
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Standard cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        main_loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        
        # If we have a bias model, add debiasing term
        if self.bias_model is not None and model.training:
            with torch.no_grad():
                # Get bias model predictions
                bias_outputs = self.bias_model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask")
                )
                bias_logits = bias_outputs.logits
            
            # Debiasing: penalize agreement with bias model
            # Use KL divergence to push distributions apart
            main_probs = F.log_softmax(logits, dim=-1)
            bias_probs = F.softmax(bias_logits, dim=-1)
            
            # We want to minimize similarity to bias model
            # Reverse KL: encourages main model to differ from bias model
            kl_loss = F.kl_div(main_probs, bias_probs, reduction='batchmean')
            
            # Total loss: main task + penalty for being too similar to bias model
            # Negative KL because we want to maximize divergence (minimize agreement)
            loss = main_loss - self.debiasing_strength * kl_loss
        else:
            loss = main_loss
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ENSEMBLE DEBIASING FOR SNLI")
    print("="*80)
    
    # Configuration
    BIAS_MODEL_PATH = './hypothesis_only_model/checkpoint-15000'  # Adjust based on your checkpoints
    OUTPUT_DIR = './debiased_model/'
    DEBIASING_STRENGTH = 0.1  # How much to penalize agreement with bias model (reduced from 0.3)
    
    # Check if bias model exists
    if not os.path.exists(BIAS_MODEL_PATH):
        print(f"\n⚠️  Bias model not found at {BIAS_MODEL_PATH}")
        print("Searching for available checkpoints...")
        
        checkpoint_dir = './hypothesis_only_model/'
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                # Use the last checkpoint
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                BIAS_MODEL_PATH = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"✓ Using checkpoint: {BIAS_MODEL_PATH}")
            else:
                print("❌ No checkpoints found. Please train hypothesis-only model first.")
                return
        else:
            print("❌ Hypothesis-only model directory not found.")
            print("Please run analyze_hypothesis_only.py first.")
            return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print(f"Debiasing strength: {DEBIASING_STRENGTH}")
    print(f"Bias model path: {BIAS_MODEL_PATH}")
    
    # Load dataset
    print("\nLoading SNLI dataset...")
    dataset = datasets.load_dataset('snli')
    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Eval examples: {len(eval_dataset)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    
    # Prepare datasets (normal, with both premise and hypothesis)
    print("\n" + "-"*80)
    print("Preparing datasets for debiased training")
    print("-"*80)
    
    def tokenize_full(examples):
        tokenized = tokenizer(
            examples['premise'],
            examples['hypothesis'],
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        tokenized['label'] = examples['label']
        return tokenized
    
    train_dataset_tokenized = train_dataset.map(
        tokenize_full,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset_tokenized = eval_dataset.map(
        tokenize_full,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    print(f"✓ Tokenized {len(train_dataset_tokenized)} training examples")
    print(f"✓ Tokenized {len(eval_dataset_tokenized)} evaluation examples")
    
    # Initialize debiased model
    print("\n" + "-"*80)
    print("Initializing debiased model")
    print("-"*80)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'google/electra-small-discriminator',
        num_labels=3
    )
    
    print("✓ Model initialized")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=500,
        eval_strategy='epoch',  # Changed from evaluation_strategy
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=2,
    )
    
    # Initialize debiased trainer
    trainer = DebiasedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_dataset_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
        bias_model_path=BIAS_MODEL_PATH,
        debiasing_strength=DEBIASING_STRENGTH,
    )
    
    # Train
    print("\n" + "="*80)
    print("TRAINING DEBIASED MODEL")
    print("="*80)
    print("\nThis will take 2-4 hours depending on your GPU...")
    print("The model is learning to focus on premise-hypothesis interaction")
    print("rather than hypothesis-only patterns.\n")
    
    trainer.train()
    
    # Save final model
    trainer.save_model()
    print(f"\n✓ Debiased model saved to {OUTPUT_DIR}")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATING DEBIASED MODEL")
    print("="*80)
    
    results = trainer.evaluate()
    
    print("\n📊 Debiased Model Results:")
    print(f"  Accuracy: {results['eval_accuracy']*100:.2f}%")
    
    # Save results
    with open(f'{OUTPUT_DIR}/eval_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions for further analysis
    predictions = trainer.predict(eval_dataset_tokenized)
    
    with open(f'{OUTPUT_DIR}/eval_predictions.jsonl', 'w') as f:
        for i, example in enumerate(eval_dataset):
            example_with_prediction = dict(example)
            example_with_prediction['predicted_scores'] = predictions.predictions[i].tolist()
            example_with_prediction['predicted_label'] = int(predictions.predictions[i].argmax())
            f.write(json.dumps(example_with_prediction))
            f.write('\n')
    
    print(f"\n✓ Predictions saved to {OUTPUT_DIR}/eval_predictions.jsonl")
    
    print("\n" + "="*80)
    print("✓ DEBIASED MODEL TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()