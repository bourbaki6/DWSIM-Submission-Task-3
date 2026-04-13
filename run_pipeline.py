


import argparse
import subprocess
import sys
import time
 
STEPS = [
    ("sampling", "data_generation.py"),
    ("simulation", "automate.py"),
    ("preprocessing", "preprocessing.py"),
    ("training", "train_models.py"),
    ("evaluation", "evaluate.py"),
]
 
STEP_NAMES = [s[0] for s in STEPS]
 
 
def parse_args():

    parser = argparse.ArgumentParser(
        description ="Run the full distillation surrogate modelling pipeline."
    )
    parser.add_argument(
        "--from", dest = "from_step", metavar = "STEP",
        choices = STEP_NAMES, default = None,
        help = "Start the pipeline from this step (skips earlier steps).",
    )
    parser.add_argument(
        "--only", dest = "only_step", metavar = "STEP",
        choices = STEP_NAMES, default = None,
        help = "Run only this single step.",
    )
    parser.add_argument(
        "--skip", dest = "skip_steps", nargs = "*", metavar = "STEP",
        choices = STEP_NAMES, default = [],
        help = "Skip one or more steps.",
    )
    return parser.parse_args()

def run_step(name: str, script: str) -> bool:

    print(f"  STEP: {name.upper()}  ({script})")

    t0 = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
 
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed (exit code {result.returncode})")
        print("Pipeline stopped. Fix the error above and re-run.")
        return False
 
    print(f"\n  Completed in {elapsed:.1f}s")
    
    return True

def select_steps(args) -> list:

    steps = list(STEPS)
 
    if args.only_step:
        steps = [(n, s) for n, s in steps if n == args.only_step]
        return steps
 
    if args.from_step:
        from_idx = STEP_NAMES.index(args.from_step)
        steps = steps[from_idx:]
 
    if args.skip_steps:
        steps = [(n, s) for n, s in steps if n not in args.skip_steps]
 
    return steps

def main():
    args = parse_args()
    steps = select_steps(args)
 
    if not steps:
        print("No steps selected. Nothing to run.")
        return
    
    print("\n Distillation Surrogate Model — Pipeline")
    
    print(f"  Steps to run: {[n for n,_ in steps]}")

    total_start = time.time()
 
    for name, script in steps:
        ok = run_step(name, script)
        if not ok:
            sys.exit(1)
 
    total = time.time() - total_start
    
    print(f"  Pipeline complete in {total:.1f}s")
    
    print("\n  Key outputs:")
    print(" dataset.csv — simulation dataset")
    print(" results_metrics.csv  — model comparison table")
    print(" results/sample_predictions.csv — example predictions")
    print(" results/trend_validation.csv  — physical trend check")
    print(" plots/  — all figures")
    print(" models/ — saved model files")
    print("\n  Launch the app:  streamlit run app.py")
 
 
if __name__ == "__main__":
    main()