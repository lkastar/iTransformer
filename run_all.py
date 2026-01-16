import argparse
import os
import subprocess
import glob

def main():
    parser = argparse.ArgumentParser(description="Recursively run training scripts.")
    parser.add_argument("--dir", required=False, default="scripts/multivariate_forecasting", help="Target directory to search scripts")
    parser.add_argument("--model", required=False, default="Testformer", help="Model name pattern to match (files must be *<pattern>*.sh)")
    parser.add_argument("--desc", required=False, default="Baseline", help="Description for the run")
    parser.add_argument("--notes", required=False, default="Test", help="Notes for wandb")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    print(f"Searching for scripts matching '*{args.model}*.sh' in {args.dir}...")

    scripts_to_run = []
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".sh") and args.model in file:
                scripts_to_run.append(os.path.join(root, file))
    
    scripts_to_run.sort()

    if not scripts_to_run:
        print("No scripts found.")
        return

    for script in scripts_to_run:
        cmd = ["sh", script, args.desc, args.notes]
        if args.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        else:
            print(f"Running {script}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}: {e}")
                exit(1)

if __name__ == "__main__":
    main()
