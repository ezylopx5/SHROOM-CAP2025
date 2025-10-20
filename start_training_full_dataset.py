#!/usr/bin/env python3
"""
🚀 START TRAINING - XLM-RoBERTa-Large on Full 1.18M Dataset
=============================================================

Quick launcher for professor demo.

Run from SHROOM directory:
    cd /Users/haxx_sh/Desktop/JCNLP2025/SHROOM2025/SHROOM
    python3 start_training_full_dataset.py
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     XLM-RoBERTa-Large Training on 1.18M Balanced Dataset         ║
║                 For Professor Demo Tomorrow!                     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Verify we're in the right directory
    current_dir = Path.cwd()
    script_dir = Path(__file__).parent
    
    print(f"📂 Current directory: {current_dir}")
    print(f"📂 Script directory: {script_dir}")
    print("")
    
    # Check if train_xlm.py exists
    train_script = script_dir / "train_xlm.py"
    if not train_script.exists():
        print(f"❌ Error: train_xlm.py not found at {train_script}")
        print(f"   Please run this script from: {script_dir}")
        sys.exit(1)
    
    print("✅ Found train_xlm.py")
    print("")
    
    # Dataset information
    print("📊 Dataset Statistics:")
    print("   - Total samples: 1,185,458")
    print("   - Balance: 50% CORRECT / 50% HALLUCINATED")
    print("   - Sources: hallucination_dataset_100k, LibreEval, FactCHD, SHROOM")
    print("")
    print("🤖 Model: XLM-RoBERTa-Large (355M parameters)")
    print("⏱️  Expected training time: 2-3 days")
    print("💾 Checkpoints: Every 5,000 steps (~2-3 hours)")
    print("🎯 Expected F1 score: 0.88-0.92")
    print("🏆 Expected rank: Top 2-3!")
    print("")
    
    # Data directory (relative to SHROOM folder)
    data_dir = "SHROOM_DATA"
    output_dir = "outputs_xlm_large_full_dataset"
    
    # Training command
    command = [
        sys.executable,  # Use current Python interpreter
        str(train_script),
        "--model_name", "xlm-roberta-large",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--epochs", "3",
        "--batch_size", "16",
        "--learning_rate", "2e-5",
        "--max_length", "256",
        "--weight_decay", "0.01",
        "--warmup_ratio", "0.1",
        
        # ✅ Checkpoint saving (every 5K steps ~ every 2-3 hours)
        "--save_steps", "5000",
        "--eval_steps", "2500",
        "--logging_steps", "100",
        "--save_total_limit", "3",  # Keep last 3 checkpoints
        
        "--n_folds", "1",  # Single run for speed
    ]
    
    print("📋 Training Command:")
    print("   " + " \\\n      ".join([str(c) for c in command]))
    print("")
    
    response = input("🚀 Start training now? This will take 2-3 days. [y/N]: ")
    if response.lower() != 'y':
        print("❌ Training cancelled.")
        sys.exit(0)
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING...")
    print("="*70)
    print("")
    print("💡 TIPS:")
    print("   - Training runs in foreground (keep terminal open)")
    print("   - Checkpoints saved every 2-3 hours in:")
    print(f"     {output_dir}/fold_0/checkpoint-XXXXX/")
    print("   - Monitor progress:")
    print(f"     tail -f {output_dir}/fold_0/training.log")
    print("   - If interrupted, just run this script again (auto-resumes!)")
    print("")
    print("="*70)
    print("")
    
    try:
        # Run training in the script directory
        os.chdir(script_dir)
        result = subprocess.run(command, check=True)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"""
🎉 Next Steps for Professor Demo:

1. Check final model:
   ls -lh {output_dir}/fold_0/best_model/

2. Generate predictions:
   python3 generate_final_submission.py \\
     --model_path {output_dir}/fold_0/best_model \\
     --data_dir {data_dir} \\
     --output_dir submissions

3. Validate format:
   cd ../..  # Go to project root
   python3 validate_submissions.py
   python3 check_submission_format.py

4. Show professor:
   - Training logs: {output_dir}/fold_0/training.log
   - F1 score: Check final validation metrics
   - Predictions: submissions/*.jsonl files

5. Submit to competition:
   - Upload to: https://shroomcap.pythonanywhere.com/
   - Get leaderboard rank!
        """)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        print("Check logs for details.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted!")
        print("Progress saved in checkpoints. Run this script again to resume.")
        sys.exit(130)


if __name__ == "__main__":
    main()
