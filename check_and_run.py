#!/usr/bin/env python3

"""
Automation Wrapper for Redrob Candidate Ranker
Handles environment checking, Stage 1 Precomputation, and Stage 2 Ranking sequentially.
"""

import os
import sys
import subprocess
import json
from datetime import datetime


def check_environment():
    """Verify all required dependencies are installed."""
    print("[*] Checking workspace packages and dependencies...")
    required_packages = ["numpy", "pandas", "yaml", "sentence_transformers", "torch"]
    missing_packages = []
    
    for pkg in required_packages:
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
        except ImportError:
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"[-] Missing dependencies detected: {missing_packages}")
        print("[*] Installing required packages via pip...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "numpy", "pandas", "pyyaml", "sentence-transformers", "torch"
            ])
            print("[+] All dependencies installed successfully.")
        except Exception as e:
            print(f"[-] Automated installation failed: {e}")
            print("Please run 'pip install -r requirements.txt' manually.")
            sys.exit(1)
    else:
        print("[+] Environment check passed. All dependencies are available.")


def locate_dataset():
    """Search for candidate dataset file in workspace."""
    print("[*] Locating candidate pool dataset...")
    possible_paths = [
        "candidates.jsonl.gz",
        "candidates.jsonl",
        "../candidates.jsonl.gz",
        "../candidates.jsonl"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[+] Found dataset file at: {path}")
            return path
    
    print("[-] Error: Could not locate candidates.jsonl or candidates.jsonl.gz in the workspace.")
    print("Please place your dataset file in the repository root directory.")
    sys.exit(1)


def run_pipeline(dataset_path):
    """Execute the complete two-stage ranking pipeline."""
    print("\n" + "="*80)
    print("RUNNING AUTOMATED TWO-STAGE CANDIDATE RANKING PIPELINE")
    print("="*80)
    
    # Stage 1: Precomputation
    print("\n[*] Launching Stage 1: Precomputation, Honeypot Detection & Embedding...")
    try:
        # Use sys.executable to ensure we use the exact same Python environment
        subprocess.check_call([
            sys.executable, "-m", "src.precompute",
            "--input", dataset_path,
            "--output", "precomputed_pool.npz"
        ])
        print("[+] Stage 1 Precomputation completed successfully. Index cached.")
    except subprocess.CalledProcessError as e:
        print(f"[-] Stage 1 critical execution error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[-] Stage 1 unexpected error: {e}")
        sys.exit(1)
    
    # Stage 2: Fast Inference & Ranking
    print("\n[*] Launching Stage 2: Fast Matrix Inference & 23 Behavioral Multipliers...")
    try:
        subprocess.check_call([
            sys.executable, "rank.py",
            "--candidates", "precomputed_pool.npz",
            "--output", "punjab.csv"
        ])
        print("\n[🎉] PIPELINE EXECUTION SUCCESSFUL!")
        print("[+] Your final submission spreadsheet 'punjab.csv' has been generated.")
        print("\nOutput Location: ./punjab.csv")
        print("Top 100 ranked candidates ready for submission!")
    except subprocess.CalledProcessError as e:
        print(f"[-] Stage 2 critical ranking execution error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[-] Stage 2 unexpected error: {e}")
        sys.exit(1)


def main():
    """Main orchestration function."""
    print("="*80)
    print("REDROB CHALLENGE - COMPLIANCE & AUTOMATION CHECKER")
    print("="*80)
    print(f"[*] Execution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    check_environment()
    dataset_file = locate_dataset()
    run_pipeline(dataset_file)


if __name__ == "__main__":
    main()
