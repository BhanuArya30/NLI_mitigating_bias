"""
Master Script: Complete Part 2 Pipeline
========================================
This script runs all Part 2 steps in the correct order.

Steps:
1. Train debiased model (2-4 hours)
2. Evaluate debiased model vs baseline
3. Test hypothesis-only performance
4. Generate all comparison visualizations
5. Create summary report

Usage:
    python3 run_part2_pipeline.py

Or run steps individually:
    python3 run_part2_pipeline.py --step train
    python3 run_part2_pipeline.py --step evaluate
    python3 run_part2_pipeline.py --step test_hyp_only
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import argparse

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def print_step(step_num, total_steps, description):
    """Print step information"""
    print(f"\n{'─'*80}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'─'*80}\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {command}\n")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed/60:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running command: {e}")
        print(f"   Command: {command}")
        return False

def check_prerequisites():
    """Check if prerequisites are met"""
    print_header("CHECKING PREREQUISITES")
    
    required_files = [
        './trained_model/',  # Baseline model
        './hypothesis_only_model/',  # Hypothesis-only model
        './eval_output/eval_predictions.jsonl',  # Baseline predictions
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        print("✗ Missing required files:")
        for path in missing:
            print(f"   - {path}")
        print("\nPlease ensure you have:")
        print("  1. Trained baseline model (./trained_model/)")
        print("  2. Trained hypothesis-only model (./hypothesis_only_model/)")
        print("  3. Baseline predictions (./eval_output/)")
        return False
    
    print("✓ All prerequisites found")
    return True

def step_train_debiased():
    """Step 1: Train debiased model"""
    print_step(1, 4, "Training Debiased Model")
    
    print("This will take 2-4 hours depending on your GPU.")
    print("The model will be saved to: ./debiased_model/")
    print("\nStarting training...\n")
    
    success = run_command(
        "python3 ./train_debiased_model.py",
        "Train debiased model"
    )
    
    if success:
        print("\n✓ Debiased model training complete!")
        print("   Model saved to: ./debiased_model/")
    else:
        print("\n✗ Training failed. Check error messages above.")
    
    return success

def step_evaluate():
    """Step 2: Evaluate debiased model"""
    print_step(2, 4, "Evaluating Debiased Model")
    
    print("Comparing baseline vs debiased performance...")
    print("This will generate comparison plots and metrics.\n")
    
    success = run_command(
        "python3 ./evaluate_debiased_model.py",
        "Evaluate debiased model"
    )
    
    if success:
        print("\n✓ Evaluation complete!")
        print("   Results saved to: ./comparison_plots/")
    else:
        print("\n✗ Evaluation failed. Check error messages above.")
    
    return success

def step_test_hypothesis_only():
    """Step 3: Test hypothesis-only performance"""
    print_step(3, 4, "Testing Hypothesis-Only Performance")
    
    print("Testing if debiasing reduced hypothesis-only accuracy...")
    print("This is the KEY test to prove debiasing worked!\n")
    
    success = run_command(
        "python3 ./test_hypothesis_only_debiased.py",
        "Test hypothesis-only performance"
    )
    
    if success:
        print("\n✓ Hypothesis-only test complete!")
        print("   Results saved to: ./hypothesis_only_comparison.json")
        print("   Visualization: ./hypothesis_only_comparison.png")
    else:
        print("\n✗ Test failed. Check error messages above.")
    
    return success

def step_summary():
    """Step 4: Generate summary"""
    print_step(4, 4, "Generating Summary Report")
    
    print("Creating comprehensive summary of all results...\n")
    
    # Read key results
    try:
        import json
        
        # Load hypothesis-only comparison
        with open('./hypothesis_only_comparison.json', 'r') as f:
            hyp_only = json.load(f)
        
        # Load evaluation summary
        with open('./comparison_plots/EVALUATION_SUMMARY.txt', 'r') as f:
            eval_summary = f.read()
        
        # Create consolidated report
        report = f"""
{'='*80}
PART 2: DEBIASING RESULTS SUMMARY
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'─'*80}
KEY FINDING: HYPOTHESIS-ONLY BIAS REDUCTION
{'─'*80}

Baseline (hypothesis-only):  {hyp_only['baseline_hypothesis_only']:.2f}%
Debiased (hypothesis-only):  {hyp_only['debiased_hypothesis_only']:.2f}%
Reduction:                   {hyp_only['reduction']:.2f} percentage points

{'✓ SUCCESS' if hyp_only['reduction'] > 5 else '~ PARTIAL' if hyp_only['reduction'] > 0 else '✗ LIMITED'}: {'Major' if hyp_only['reduction'] > 10 else 'Moderate' if hyp_only['reduction'] > 5 else 'Minimal'} reduction in hypothesis bias

{'─'*80}
DETAILED EVALUATION RESULTS
{'─'*80}

{eval_summary}

{'─'*80}
GENERATED FILES
{'─'*80}

Models:
  ./debiased_model/                      - Trained debiased model
  ./debiased_model/eval_predictions.jsonl - Model predictions

Evaluation Results:
  ./comparison_plots/                     - Comparison visualizations
  ./comparison_plots/EVALUATION_SUMMARY.txt - Detailed metrics
  ./hypothesis_only_comparison.json       - Bias reduction results
  ./hypothesis_only_comparison.png        - Bias reduction chart

Visualizations for Report:
  1. comparison_plots/1_overall_comparison.png
  2. comparison_plots/2_per_label_comparison.png
  3. comparison_plots/3_overlap_stratified.png
  4. comparison_plots/4_prediction_changes.png
  5. hypothesis_only_comparison.png (MOST IMPORTANT!)

{'─'*80}
NEXT STEPS FOR REPORT WRITING
{'─'*80}

1. Review all generated visualizations
2. Examine sample improved/degraded examples in:
   - ./comparison_plots/EVALUATION_SUMMARY.txt
3. Check hypothesis-only reduction (key metric!)
4. Write Part 2 section using PART2_SUMMARY.md as guide
5. Include key visualizations in report
6. Discuss why results occurred (analysis!)

{'─'*80}
INTERPRETATION GUIDE
{'─'*80}

Hypothesis-Only Reduction:
  > 10 points  = Excellent debiasing
  5-10 points  = Good debiasing
  0-5 points   = Partial debiasing
  < 0 points   = Ineffective (discuss why!)

Overall Accuracy:
  Within 2% of baseline = Good (acceptable tradeoff)
  2-4% decrease = Moderate cost
  > 4% decrease = High cost (discuss if worth it)

Low-Overlap Performance:
  Improvement = Success on hard cases
  No change = Artifacts remain
  Decrease = May have traded artifacts

{'='*80}
"""
        
        # Save report
        with open('./PART2_COMPLETE_SUMMARY.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print("✓ Summary report saved to: ./PART2_COMPLETE_SUMMARY.txt")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating summary: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Part 2 pipeline')
    parser.add_argument('--step', type=str, choices=['train', 'evaluate', 'test_hyp_only', 'summary', 'all'],
                       default='all', help='Which step to run')
    
    args = parser.parse_args()
    
    print_header("PART 2: DEBIASING PIPELINE")
    
    print(f"Pipeline mode: {args.step.upper()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n✗ Prerequisites not met. Please complete Part 1 first.")
        return 1
    
    total_start = time.time()
    
    # Run requested steps
    if args.step == 'all':
        steps = [
            ('train', step_train_debiased),
            ('evaluate', step_evaluate),
            ('test_hyp_only', step_test_hypothesis_only),
            ('summary', step_summary)
        ]
    else:
        step_map = {
            'train': step_train_debiased,
            'evaluate': step_evaluate,
            'test_hyp_only': step_test_hypothesis_only,
            'summary': step_summary
        }
        steps = [(args.step, step_map[args.step])]
    
    # Execute steps
    failed_steps = []
    for step_name, step_func in steps:
        success = step_func()
        if not success:
            failed_steps.append(step_name)
            if args.step == 'all':
                print(f"\n⚠️  Step '{step_name}' failed. Stopping pipeline.")
                break
    
    # Final summary
    total_elapsed = time.time() - total_start
    
    print_header("PIPELINE COMPLETE")
    
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_steps:
        print(f"\n✗ Failed steps: {', '.join(failed_steps)}")
        return 1
    else:
        print("\n✓ All steps completed successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())