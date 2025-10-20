#!/usr/bin/env python3
"""
Generate SHROOM-CAP Competition Submission Files

This script:
1. Loads test data from SHROOM_DATA/Test/
2. Loads our best model (XLM-RoBERTa trained with augmented data)
3. Makes predictions
4. Generates submission files in the correct format:
   - One .jsonl file per language
   - Format: {"index": "lang-tst-ID", "has_factual_mistakes": "y/n", "has_fluency_mistakes": "n"}
   - Order must match original test data

Usage:
    python generate_final_submission.py --model_path outputs_xlm_english/fold_0/best_model --data_dir SHROOM_DATA
"""

import argparse
import json
import logging
from pathlib import Path
import os
import sys

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SHROOMTestDataset(Dataset):
    """Dataset for SHROOM test data"""
    
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Create input text: "Premise: ... Hypothesis: ..."
        # Use same format as training
        premise = str(row.get('src', row.get('abstract', '')))
        hypothesis = str(row.get('hyp', row.get('output_text', '')))
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
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'index': row['index'],  # Keep original index
            'language': row['language']
        }


def load_test_data(data_dir: str) -> pd.DataFrame:
    """
    Load test data from SHROOM_DATA/Test/
    
    Returns DataFrame with columns: id, abstract, output_text, language, index
    """
    logger.info(f"📂 Loading test data from: {data_dir}")
    
    data_path = Path(data_dir)
    test_path = data_path / "Test"
    
    if not test_path.exists():
        logger.error(f"❌ Test directory not found: {test_path}")
        sys.exit(1)
    
    all_test_data = []
    
    # All test languages (including zero-shot Indic languages not in training)
    # Training: en, es, fr, hi, it
    # Test-only (zero-shot): bn, gu, ml, te
    languages = ['bn', 'en', 'es', 'fr', 'gu', 'hi', 'it', 'ml', 'te']
    
    for lang in languages:
        # Look for test file patterns
        test_file = test_path / f"{lang}_test_data.jsonl"
        
        if not test_file.exists():
            logger.warning(f"⚠️  Test file not found: {test_file}")
            continue
        
        # Load test data
        df = pd.read_json(test_file, lines=True)
        df['language'] = lang
        
        # The test data already has the 'index' field in the format we need
        # Just verify that it exists and is properly formatted
        if 'index' not in df.columns:
            logger.error(f"❌ {lang}: 'index' column is missing in the test data")
            continue
        
        # Verify index format (should be like "en-tst-123")
        sample_index = df['index'].iloc[0] if len(df) > 0 else ""
        if not sample_index.startswith(f"{lang}-tst-"):
            logger.warning(f"⚠️  {lang}: Index format may be incorrect: {sample_index}")
            # We'll keep the original index anyway
        
        all_test_data.append(df)
        logger.info(f"   ✅ {lang}: {len(df)} samples")
    
    if not all_test_data:
        logger.error("❌ No test data loaded!")
        sys.exit(1)
    
    # Combine all languages
    test_df = pd.concat(all_test_data, ignore_index=True)
    logger.info(f"✅ Total test samples: {len(test_df)}")
    
    # Check required columns
    if 'src' not in test_df.columns:
        if 'abstract' in test_df.columns:
            test_df['src'] = test_df['abstract']
            logger.info("   Mapped 'abstract' -> 'src'")
    
    if 'hyp' not in test_df.columns:
        if 'output_text' in test_df.columns:
            test_df['hyp'] = test_df['output_text']
            logger.info("   Mapped 'output_text' -> 'hyp'")
    
    required_columns = ['src', 'hyp', 'index', 'language']
    for col in required_columns:
        if col not in test_df.columns:
            logger.error(f"❌ Required column missing: {col}")
            sys.exit(1)
    
    return test_df


def load_model(model_path: str, device: str = 'cpu'):
    """
    Load trained model
    """
    logger.info(f"🔄 Loading model from: {model_path}")
    
    try:
        # Load model (it includes the tokenizer config)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Try to load tokenizer from model path first, fallback to xlm-roberta-large
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"   Using tokenizer from model checkpoint")
        except:
            base_model = "xlm-roberta-large"
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            logger.info(f"   Using tokenizer from: {base_model}")
        
        logger.info(f"   Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"✅ Model loaded on {device}")
    
    return model, tokenizer


def predict(
    model,
    tokenizer,
    test_data: pd.DataFrame,
    batch_size: int = 32,
    device: str = 'cpu'
) -> dict:
    """
    Make predictions on test data
    
    Returns: Dictionary with index -> prediction mapping
    """
    logger.info(f"🔮 Making predictions on {len(test_data)} samples...")
    
    dataset = SHROOMTestDataset(test_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            indices = batch['index']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Map indices to predictions
            for idx, pred in zip(indices, batch_preds):
                predictions[idx] = int(pred)
    
    logger.info(f"✅ Predictions complete")
    
    return predictions


def generate_submissions(
    test_data: pd.DataFrame,
    predictions: dict,
    output_dir: str = 'submissions'
):
    """
    Generate submission files for each language
    
    Format:
    {
        "index": "lang-tst-ID",
        "has_factual_mistakes": "y/n",
        "has_fluency_mistakes": "n"
    }
    """
    logger.info(f"📝 Generating submission files...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add predictions to dataframe
    test_data['prediction'] = test_data['index'].map(predictions)
    
    # Generate submissions for each language
    languages = test_data['language'].unique()
    
    for lang in languages:
        lang_data = test_data[test_data['language'] == lang].copy()
        
        # Sort by numeric ID (e.g., en-tst-0, en-tst-1, en-tst-2, ...)
        # Extract number from index like "en-tst-123" and sort numerically
        lang_data['_sort_key'] = lang_data['index'].apply(
            lambda x: int(x.split('-tst-')[1]) if '-tst-' in x else 0
        )
        lang_data = lang_data.sort_values('_sort_key').reset_index(drop=True)
        
        submission_file = os.path.join(output_dir, f"{lang}_submission.jsonl")
        
        with open(submission_file, 'w', encoding='utf-8') as f:
            for _, row in lang_data.iterrows():
                # Convert prediction to y/n format
                has_mistakes = "y" if row['prediction'] == 1 else "n"
                
                submission_obj = {
                    "index": row['index'],
                    "has_factual_mistakes": has_mistakes,
                    "has_fluency_mistakes": "n"
                }
                
                f.write(json.dumps(submission_obj, ensure_ascii=False) + '\n')
        
        logger.info(f"   ✅ {lang}: {len(lang_data)} predictions -> {submission_file}")
        
        # Show prediction distribution
        dist = lang_data['prediction'].value_counts()
        pct_hallucination = (dist.get(1, 0) / len(lang_data)) * 100
        logger.info(f"      {lang} hallucination rate: {pct_hallucination:.1f}% ({dist.get(1, 0)}/{len(lang_data)})")
    
    logger.info(f"✅ All submission files saved to: {output_dir}")
    logger.info(f"📦 Ready for competition upload!")


def main():
    parser = argparse.ArgumentParser(description='Generate SHROOM-CAP submission files')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='outputs_xlm_english/fold_0/best_model',
                       help='Path to trained model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='SHROOM_DATA',
                       help='Path to SHROOM_DATA directory')
    parser.add_argument('--output_dir', type=str, default='submissions',
                       help='Output directory for submission files')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu',
                       help='Device: mps, cuda, or cpu')
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Model path not found: {args.model_path}")
        sys.exit(1)
    
    # Load test data
    test_data = load_test_data(args.data_dir)
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.device)
    
    # Make predictions
    predictions = predict(model, tokenizer, test_data, args.batch_size, args.device)
    
    # Generate submission files
    generate_submissions(test_data, predictions, args.output_dir)


if __name__ == "__main__":
    main()