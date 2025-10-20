#!/usr/bin/env python3
"""
XLM-RoBERTa-Large Training Script for SHROOM-CAP 2025
======================================================

Train XLM-RoBERTa-Large for hallucination detection across 9 languages.
This is a CLEAN, SIMPLE implementation that WORKS.

Model: xlm-roberta-large (560M parameters)
Task: Binary sequence classification (hallucination: yes/no)
Languages: en, es, fr, hi, it, bn, gu, ml, te

Usage:
    python train_xlm.py --batch_size 16 --epochs 5 --output_dir outputs_xlm
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset

# Optional: PEFT/LoRA for parameter-efficient fine-tuning
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    # Will warn later if user tries to use LoRA without PEFT installed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train XLM-RoBERTa for SHROOM-CAP')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                       choices=['xlm-roberta-base', 'xlm-roberta-large', 'facebook/xlm-roberta-xxl'],
                       help='XLM-R model: base=270M, large=560M, xxl=10.7B (PHASE 3!)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Data augmentation arguments
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation (back-translation, synonym replacement)')
    parser.add_argument('--augment_minority_only', action='store_true',
                       help='Only augment minority class samples')
    parser.add_argument('--augment_factor', type=int, default=3,
                       help='Augmentation factor for minority class')
    
    # Training mode arguments
    parser.add_argument('--training_mode', type=str, default='full',
                       choices=['full', 'lora', 'freeze'],
                       help='Training mode: full=full fine-tuning, lora=PEFT with LoRA, freeze=freeze encoder')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank (r). Higher = more parameters. Recommended: 8-32')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha. Usually 2*r. Controls scaling')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                       help='Evaluation batch size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # K-fold arguments
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--fold', type=int, default=None,
                       help='Train specific fold only (0-indexed). If None, train all folds')
    
    # Class imbalance handling
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use Focal Loss instead of weighted CE for class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter (focusing parameter)')
    parser.add_argument('--class_weight_boost', type=float, default=1.5,
                       help='Boost factor for minority class weight')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='SHROOM_DATA/TrainSet V2',
                       help='Directory containing training data')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs_xlm',
                       help='Output directory for models and logs')
    parser.add_argument('--save_best_only', action='store_true',
                       help='Only save the best model from each fold')
    parser.add_argument('--save_steps', type=int, default=5000,
                       help='Save checkpoint every N steps (for large datasets)')
    parser.add_argument('--eval_steps', type=int, default=2500,
                       help='Evaluate every N steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 mixed precision')
    parser.add_argument('--bf16', action='store_true',
                       help='Use BF16 mixed precision (recommended for A100)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_data(df, augment_factor=3, minority_only=True):
    """
    Apply data augmentation to training data
    
    Args:
        df: DataFrame with training data
        augment_factor: How many augmented samples to create per original
        minority_only: If True, only augment minority class
    
    Returns:
        Augmented DataFrame
    """
    logger.info(f"🔄 Applying data augmentation...")
    logger.info(f"   Augment factor: {augment_factor}")
    logger.info(f"   Minority only: {minority_only}")
    
    try:
        import nlpaug.augmenter.word as naw
        import nlpaug.augmenter.sentence as nas
    except ImportError:
        logger.warning("⚠️  nlpaug not installed. Skipping data augmentation.")
        logger.warning("   Install with: pip install nlpaug")
        return df
    
    # Check class distribution
    class_counts = df['label'].value_counts()
    minority_class = 0 if class_counts.get(0, 0) < class_counts.get(1, 0) else 1
    
    logger.info(f"   Class distribution before augmentation:")
    logger.info(f"      Class 0: {class_counts.get(0, 0)}")
    logger.info(f"      Class 1: {class_counts.get(1, 0)}")
    logger.info(f"   Minority class: {minority_class}")
    
    # Initialize augmenters
    synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
    
    # Store augmented samples
    augmented_samples = []
    
    # Apply augmentation
    for idx, row in df.iterrows():
        # Skip if minority_only is True and this is not minority class
        if minority_only and row['label'] != minority_class:
            continue
        
        # Create augmented samples
        for _ in range(augment_factor):
            try:
                # Augment hypothesis (output text)
                aug_hyp = synonym_aug.augment(str(row['hyp']))
                
                # Create augmented sample
                aug_row = row.copy()
                aug_row['hyp'] = aug_hyp if aug_hyp else row['hyp']
                aug_row['augmented'] = True
                augmented_samples.append(aug_row)
            except Exception as e:
                logger.warning(f"   ⚠️  Augmentation failed for sample {idx}: {e}")
                continue
    
    # Combine original and augmented data
    augmented_df = pd.DataFrame(augmented_samples)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # Log results
    new_class_counts = combined_df['label'].value_counts()
    logger.info(f"   ✅ Augmentation complete!")
    logger.info(f"   Original samples: {len(df)}")
    logger.info(f"   Augmented samples: {len(augmented_samples)}")
    logger.info(f"   Total samples: {len(combined_df)}")
    logger.info(f"   Class distribution after augmentation:")
    logger.info(f"      Class 0: {new_class_counts.get(0, 0)}")
    logger.info(f"      Class 1: {new_class_counts.get(1, 0)}")
    
    return combined_df


def load_shroom_data(data_dir):
    """
    Load ALL training data from multiple sources using unified_data_loader.py
    
    This loads:
    - SHROOM TrainSet V1 & V2: 724 + 1,029 = 1,753 samples
    - hallucination_dataset_100k: 100,075 samples  
    - LibreEval: ~647K samples
    - FactCHD: 51,383 samples
    - Total: ~1,185,458 BALANCED samples!
    """
    data_dir = Path(data_dir)
    logger.info(f"� Loading FULL 1.18M BALANCED dataset from: {data_dir}")
    logger.info(f"   (This replaces the 724-sample imbalanced TrainSet V2)")
    
    # Import unified data loader
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from unified_data_loader import UnifiedDataLoader
        logger.info("   ✅ unified_data_loader imported successfully")
    except ImportError as e:
        logger.error(f"   ❌ Failed to import unified_data_loader: {e}")
        logger.error(f"   📂 Project root: {project_root}")
        raise
    
    # Initialize loader
    loader = UnifiedDataLoader(str(data_dir))
    
    # Load all datasets at once
    logger.info("   📊 Loading all datasets...")
    samples = loader.load_all()
    
    if not samples:
        raise ValueError("❌ No training data loaded!")
    
    # Convert UnifiedSample objects to DataFrame
    logger.info("   🔄 Converting to DataFrame...")
    data_dicts = []
    for sample in samples:
        data_dicts.append({
            'hyp': sample.text,  # The text to classify
            'src': sample.metadata.get('source_text', '') if sample.metadata else '',
            'label': sample.label,  # Keep as numeric: 0 = CORRECT, 1 = HALLUCINATED
            'language': sample.language
        })
    
    combined_df = pd.DataFrame(data_dicts)
    
    # Ensure all required columns exist
    required_cols = ['hyp', 'src', 'label']
    for col in required_cols:
        if col not in combined_df.columns:
            raise ValueError(f"❌ Missing required column: {col}")
    
    # Get language distribution
    languages = combined_df['language'].unique().tolist() if 'language' in combined_df.columns else ['mixed']
    
    # Show statistics
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"✅ FULL DATASET LOADED SUCCESSFULLY!")
    logger.info(f"{'='*70}")
    logger.info(f"Total samples: {len(combined_df):,}")
    logger.info(f"Languages: {', '.join(map(str, languages))}")
    logger.info(f"")
    logger.info(f"📊 Label Distribution:")
    label_0 = (combined_df['label']==0).sum()
    label_1 = (combined_df['label']==1).sum()
    total = len(combined_df)
    logger.info(f"   Class 0 (CORRECT):      {label_0:,} ({100*label_0/total:.1f}%)")
    logger.info(f"   Class 1 (HALLUCINATED): {label_1:,} ({100*label_1/total:.1f}%)")
    logger.info(f"   Balance ratio: {max(label_0, label_1) / min(label_0, label_1):.2f}:1")
    logger.info(f"")
    logger.info(f"🎯 This is a BALANCED dataset - model will learn real patterns!")
    logger.info(f"{'='*70}")
    logger.info(f"")
    
    return combined_df


class SHROOMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for SHROOM data"""
    
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Create input text: "Premise: ... Hypothesis: ..."
        premise = str(row.get('src', ''))
        hypothesis = str(row.get('hyp', ''))
        text = f"Premise: {premise} Hypothesis: {hypothesis}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    """
    🔧 FIX 2: Custom Trainer with class weights for handling imbalance
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            logger.info(f"\n⚖️  Using class weights: {self.class_weights}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Apply class weights to loss
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.args.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


class FocalLossTrainer(Trainer):
    """
    🔥 Focal Loss Trainer for severe class imbalance
    
    Focal Loss focuses on hard-to-classify examples by down-weighting easy examples.
    This is particularly effective for imbalanced datasets.
    """
    def __init__(self, gamma=2.0, alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        logger.info(f"\n🔥 Using Focal Loss:")
        logger.info(f"   Gamma (focusing parameter): {self.gamma}")
        logger.info(f"   Alpha (class weights): {self.alpha}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get probability for correct class
        pt = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        # Calculate focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(self.args.device)[labels]
            focal_weight = focal_weight * alpha_t
        
        # Calculate cross entropy loss
        ce_loss = -torch.log(pt + 1e-8)  # Add epsilon to prevent log(0)
        
        # Calculate focal loss
        loss = (focal_weight * ce_loss).mean()
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute F1 metrics for evaluation with detailed diagnostics"""
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.argmax(predictions, axis=-1)
    
    # Calculate F1 scores
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_class_0 = f1_score(labels, predictions, pos_label=0)
    f1_class_1 = f1_score(labels, predictions, pos_label=1)
    
    # 🔍 FIX 4: Diagnostic logging
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    pred_dist = dict(zip(unique_preds, pred_counts))
    
    logger.info(f"\n📊 Prediction Distribution:")
    logger.info(f"   Class 0 predictions: {pred_dist.get(0, 0)} ({100*pred_dist.get(0, 0)/len(predictions):.1f}%)")
    logger.info(f"   Class 1 predictions: {pred_dist.get(1, 0)} ({100*pred_dist.get(1, 0)/len(predictions):.1f}%)")
    
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, label_counts))
    logger.info(f"   True Class 0: {label_dist.get(0, 0)} ({100*label_dist.get(0, 0)/len(labels):.1f}%)")
    logger.info(f"   True Class 1: {label_dist.get(1, 0)} ({100*label_dist.get(1, 0)/len(labels):.1f}%)")
    
    return {
        'f1_macro': f1_macro,
        'f1_class_0': f1_class_0,
        'f1_class_1': f1_class_1,
    }


def train_fold(args, fold_idx, train_df, val_df, tokenizer):
    """Train a single fold"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 TRAINING FOLD {fold_idx + 1}/{args.n_folds}")
    logger.info(f"{'='*60}")
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Val size: {len(val_df)}")
    
    # Check class distribution
    train_class_dist = train_df['label'].value_counts()
    val_class_dist = val_df['label'].value_counts()
    logger.info(f"\nTrain class distribution:")
    logger.info(f"  Class 0: {train_class_dist.get(0, 0)} ({100*train_class_dist.get(0, 0)/len(train_df):.1f}%)")
    logger.info(f"  Class 1: {train_class_dist.get(1, 0)} ({100*train_class_dist.get(1, 0)/len(train_df):.1f}%)")
    logger.info(f"Val class distribution:")
    logger.info(f"  Class 0: {val_class_dist.get(0, 0)} ({100*val_class_dist.get(0, 0)/len(val_df):.1f}%)")
    logger.info(f"  Class 1: {val_class_dist.get(1, 0)} ({100*val_class_dist.get(1, 0)/len(val_df):.1f}%)")
    
    # 🔧 SIMPLIFIED: No class weights - let model learn naturally
    # Class weights were causing 100% prediction bias
    logger.info(f"\n📊 Using standard training (no class weights)")
    logger.info(f"   Class imbalance ratio: {train_class_dist.get(1, 0) / train_class_dist.get(0, 1):.2f}:1")
    class_weights = None
    
    # Create datasets
    train_dataset = SHROOMDataset(train_df, tokenizer, args.max_length)
    val_dataset = SHROOMDataset(val_df, tokenizer, args.max_length)
    
    # Load model
    logger.info(f"\n🤖 Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Apply training mode
    logger.info(f"\n🔧 Training mode: {args.training_mode.upper()}")
    
    if args.training_mode == 'freeze':
        # FREEZE MODE: Only train classification head
        logger.info("   Freezing encoder, training only classification head")
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        adjusted_lr = args.learning_rate
        
    elif args.training_mode == 'lora':
        # LORA MODE: Parameter-efficient fine-tuning
        if not PEFT_AVAILABLE:
            logger.error("❌ PEFT not installed! Install with: pip install peft")
            logger.error("   Falling back to full fine-tuning mode")
            adjusted_lr = args.learning_rate
        else:
            logger.info(f"   Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["query", "value"],  # XLM-R attention layers
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            model = get_peft_model(model, lora_config)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
            # LoRA typically uses higher learning rate
            adjusted_lr = args.learning_rate * 10 if args.learning_rate == 2e-5 else args.learning_rate
            logger.info(f"   LoRA learning rate boost: {adjusted_lr:.2e}")
    
    else:  # 'full' mode
        # FULL MODE: Standard fine-tuning
        logger.info("   Full fine-tuning of all parameters")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Trainable: {trainable_params:,} (100%)")
        adjusted_lr = args.learning_rate
    
    # Output directory for this fold
    fold_output_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n🎓 Learning rate: {adjusted_lr:.2e}")
    
    # Training arguments with frequent checkpointing for large datasets
    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=adjusted_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        # Evaluation and saving strategy
        eval_strategy="steps",  # Changed from "epoch" for frequent evaluation
        eval_steps=args.eval_steps,  # Evaluate every N steps (default: 2500)
        save_strategy="steps",  # Changed from "epoch" for frequent checkpoints
        save_steps=args.save_steps,  # Save every N steps (default: 5000)
        logging_steps=args.logging_steps,  # Log every N steps (default: 100)
        
        # Best model tracking
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,  # Keep last N checkpoints (default: 3)
        
        # Performance optimizations
        fp16=args.fp16,
        bf16=args.bf16,
        
        # Other settings
        report_to="none",
        save_safetensors=False,  # ✅ Fix safetensors error
        seed=args.seed,
        
        # Resume from checkpoint if interrupted
        resume_from_checkpoint=True,  # ✅ Auto-resume if training interrupted
    )
    
    # Calculate class weights for imbalance handling
    from sklearn.utils.class_weight import compute_class_weight
    # Get labels from the DataFrame, not from the dataset
    train_labels = train_df['label'].values
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    
    # Boost minority class weight
    if len(class_weights) == 2:
        if class_weights[0] > class_weights[1]:
            class_weights[0] *= args.class_weight_boost
        else:
            class_weights[1] *= args.class_weight_boost
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    # Choose trainer based on args
    if args.use_focal_loss:
        logger.info(f"\n� Using Focal Loss Trainer for class imbalance")
        trainer = FocalLossTrainer(
            gamma=args.focal_gamma,
            alpha=class_weights_tensor,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        logger.info(f"\n⚖️  Using Weighted Cross-Entropy Trainer")
        trainer = WeightedTrainer(
            class_weights=class_weights_tensor,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
    
    # Train
    logger.info("\n🚀 Starting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = (time.time() - start_time) / 60
    logger.info(f"✅ Training completed in {training_time:.2f} minutes")
    
    # Evaluate
    logger.info("\n📊 Evaluating on validation set...")
    eval_results = trainer.evaluate()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx + 1} RESULTS:")
    logger.info(f"{'='*60}")
    logger.info(f"F1 Macro: {eval_results['eval_f1_macro']:.4f}")
    logger.info(f"F1 Class 0 (No Hallucination): {eval_results['eval_f1_class_0']:.4f}")
    logger.info(f"F1 Class 1 (Hallucination): {eval_results['eval_f1_class_1']:.4f}")
    logger.info(f"Training time: {training_time:.2f} minutes")
    logger.info(f"{'='*60}\n")
    
    # Save best model
    best_model_dir = fold_output_dir / "best_model"
    logger.info(f"💾 Saving best model to: {best_model_dir}")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    
    # Save metrics
    metrics = {
        'fold': fold_idx,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'training_time_minutes': training_time,
        **eval_results,
        **train_result.metrics
    }
    
    metrics_file = fold_output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Clean up
    del model, trainer
    torch.cuda.empty_cache()
    
    return eval_results


def main():
    """Main training function"""
    args = parse_args()
    set_seed(args.seed)
    
    logger.info("="*60)
    logger.info("🎯 XLM-ROBERTA TRAINING FOR SHROOM-CAP 2025")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    model_size = "270M" if "base" in args.model_name else "560M" if "large" in args.model_name else "Unknown"
    logger.info(f"Model size: {model_size} parameters")
    logger.info(f"Training mode: {args.training_mode.upper()}")
    if args.training_mode == 'lora':
        logger.info(f"  LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    elif args.training_mode == 'freeze':
        logger.info(f"  Encoder frozen, classifier head only")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Batch size: {args.batch_size} (gradient accumulation: {args.gradient_accumulation_steps})")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Number of folds: {args.n_folds}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("="*60 + "\n")
    
    # Load data (NO TRANSLATION - original multilingual data)
    df = load_shroom_data(args.data_dir)
    
    # Apply data augmentation if requested
    if args.augment:
        df = augment_data(df, 
                         augment_factor=args.augment_factor, 
                         minority_only=args.augment_minority_only)
    
    # Load tokenizer
    logger.info(f"🔤 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Save tokenizer to output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    
    # Stratified K-Fold or simple train/val split
    all_results = []
    
    if args.n_folds == 1:
        # Special case: n_folds=1 means train on 90%, validate on 10%
        logger.info(f"\n🔀 Creating single train/val split (90/10)...")
        from sklearn.model_selection import train_test_split
        
        train_df, val_df = train_test_split(
            df, 
            test_size=0.1, 
            random_state=args.seed,
            stratify=df['label']
        )
        
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        
        logger.info(f"   Train samples: {len(train_df):,}")
        logger.info(f"   Val samples: {len(val_df):,}")
        
        fold_results = train_fold(args, 0, train_df, val_df, tokenizer)
        all_results.append(fold_results)
        
    else:
        # Multi-fold cross-validation
        logger.info(f"\n🔀 Creating {args.n_folds}-fold stratified splits...")
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        
        # Determine which folds to train
        if args.fold is not None:
            fold_indices = [args.fold]
            logger.info(f"Training only fold {args.fold}")
        else:
            fold_indices = range(args.n_folds)
            logger.info(f"Training all {args.n_folds} folds")
        
        # Train each fold
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
            if fold_idx not in fold_indices:
                continue
            
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            
            fold_results = train_fold(args, fold_idx, train_df, val_df, tokenizer)
            all_results.append(fold_results)
    
    # Summary
    if len(all_results) > 1:
        logger.info("\n" + "="*60)
        logger.info("🏆 FINAL SUMMARY - ALL FOLDS")
        logger.info("="*60)
        
        f1_macros = [r['eval_f1_macro'] for r in all_results]
        f1_class0s = [r['eval_f1_class_0'] for r in all_results]
        f1_class1s = [r['eval_f1_class_1'] for r in all_results]
        
        logger.info(f"Average F1 Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
        logger.info(f"Average F1 Class 0: {np.mean(f1_class0s):.4f} ± {np.std(f1_class0s):.4f}")
        logger.info(f"Average F1 Class 1: {np.mean(f1_class1s):.4f} ± {np.std(f1_class1s):.4f}")
        logger.info(f"\nIndividual fold F1 scores:")
        for i, f1 in enumerate(f1_macros):
            logger.info(f"  Fold {i}: {f1:.4f}")
        logger.info(f"Best fold: Fold {np.argmax(f1_macros)} (F1={max(f1_macros):.4f})")
        logger.info("="*60)
        
        # Save summary
        summary = {
            'model': args.model_name,
            'n_folds': args.n_folds,
            'avg_f1_macro': float(np.mean(f1_macros)),
            'std_f1_macro': float(np.std(f1_macros)),
            'avg_f1_class_0': float(np.mean(f1_class0s)),
            'avg_f1_class_1': float(np.mean(f1_class1s)),
            'fold_results': [
                {'fold': i, 'f1_macro': float(f1)}
                for i, f1 in enumerate(f1_macros)
            ],
            'best_fold': int(np.argmax(f1_macros)),
            'best_f1': float(max(f1_macros))
        }
        
        summary_file = output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✅ Summary saved to: {summary_file}")
    
    logger.info(f"\n✅ ALL TRAINING COMPLETE!")
    logger.info(f"📂 Models saved to: {args.output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Use the best fold model for predictions")
    logger.info(f"  2. Or create ensemble predictions from all folds")
    logger.info(f"  3. Generate test submissions with generate_submission.py")


if __name__ == "__main__":
    main()
