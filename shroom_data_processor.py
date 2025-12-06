#!/usr/bin/env python3
"""
Lightning AI Data Processor for SHROOM-CAP Competition
Optimized for A100 GPU training with Google Colab backup strategy
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHROOMDataset(Dataset):
    """PyTorch Dataset for SHROOM hallucination detection"""
    
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 512, is_training: bool = True):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
            
        row = self.data.iloc[idx]
        
        # Construct input text safely
        input_text = self._build_input_text(row)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': input_text,
            'language': str(row.get('language', 'unknown'))
        }
        
        # Add labels for training data
        if self.is_training and 'label' in row and pd.notna(row['label']):
            item['labels'] = torch.tensor(int(row['label']), dtype=torch.long)
        
        # Add index if available
        if 'index' in row:
            item['index'] = str(row['index'])
        elif hasattr(row, 'name'):
            item['index'] = str(row.name)
        else:
            item['index'] = str(idx)
            
        return item
    
    def _build_input_text(self, row: pd.Series) -> str:
        """Build input text from paper and QA data with safe handling"""
        
        # Extract fields safely
        title = self._safe_str(row.get('title', ''))
        abstract = self._safe_str(row.get('abstract', ''))[:1000]  # Limit length
        question = self._safe_str(row.get('question', ''))
        output_text = self._safe_str(row.get('output_text', ''))
        authors = self._format_authors(row.get('authors', []))
        
        # Build comprehensive prompt
        input_text = f"""Paper Title: {title}

Abstract: {abstract}

Authors: {authors}

Question: {question}

Model Response: {output_text}

Task: Determine if the model response contains hallucinations (factually incorrect or unsupported information) based on the paper content."""
        
        return input_text
    
    def _safe_str(self, value: Any) -> str:
        """Safely convert value to string"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except:
            return ""
    
    def _format_authors(self, authors: Any) -> str:
        """Format authors list safely"""
        if not authors:
            return "Unknown"
            
        try:
            if isinstance(authors, str):
                return authors
            
            if isinstance(authors, list):
                formatted = []
                for author in authors[:5]:  # Limit to 5 authors
                    if isinstance(author, dict):
                        first = self._safe_str(author.get('first', ''))
                        last = self._safe_str(author.get('last', ''))
                        name = f"{first} {last}".strip()
                        if name:
                            formatted.append(name)
                    else:
                        name = self._safe_str(author).strip()
                        if name:
                            formatted.append(name)
                
                if len(authors) > 5:
                    formatted.append("et al.")
                
                return ", ".join(formatted) if formatted else "Unknown"
            
            return self._safe_str(authors)
            
        except Exception as e:
            logger.warning(f"Error formatting authors: {e}")
            return "Unknown"

class SHROOMDataProcessor:
    """Main data processor for SHROOM-CAP competition"""
    
    def __init__(self):
        self.platform = self._detect_platform()
        self.base_path = self._get_base_path()
        self.data_path = self._find_data_directory()
        self.storage_path = self.base_path / "shroom_workspace"
        self.storage_path.mkdir(exist_ok=True)
        
        self.train_data = None
        self.test_data = None
        
        logger.info(f"🔧 Platform: {self.platform}")
        logger.info(f"📁 Data path: {self.data_path}")
        logger.info(f"💾 Storage path: {self.storage_path}")
    
    def _detect_platform(self) -> str:
        """Detect current platform"""
        if os.path.exists("/teamspace/studios/this_studio"):
            return "lightning_ai"
        elif os.path.exists("/content"):
            return "colab"
        else:
            return "local"
    
    def _get_base_path(self) -> Path:
        """Get base working path for current platform"""
        if self.platform == "lightning_ai":
            return Path("/teamspace/studios/this_studio")
        elif self.platform == "colab":
            return Path("/content/drive/MyDrive/SHROOM_CAP")
        else:
            return Path(".")
    
    def _find_data_directory(self) -> Path:
        """Find SHROOM data directory"""
        possible_paths = [
            self.base_path / "SHROOM_DATA",
            self.base_path / "Data",
            self.base_path / "data",
            self.base_path
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "TrainSet V1").exists():
                logger.info(f"✅ Found data directory: {path}")
                return path
        
        # Default fallback
        default_path = self.base_path / "SHROOM_DATA"
        logger.warning(f"⚠️  Using default data path: {default_path}")
        return default_path
    
    def load_training_data(self) -> pd.DataFrame:
        """Load all training data from SHROOM dataset"""
        logger.info("📊 Loading SHROOM training data...")
        
        all_data = []
        
        # Load main training data
        train_dir = self.data_path / "TrainSet V1"
        if train_dir.exists():
            data_files = list(train_dir.glob("*_train_data.jsonl"))
            logger.info(f"Found {len(data_files)} training files: {[f.name for f in data_files]}")
            
            for data_file in data_files:
                lang = data_file.stem.split('_')[0]
                data_samples = self._load_jsonl_file(data_file)
                
                # Load corresponding labels
                label_file = train_dir / f"{lang}_train_label.jsonl"
                if label_file.exists():
                    labels = self._load_jsonl_file(label_file)
                    
                    # Combine data and labels
                    for i, (data_item, label_item) in enumerate(zip(data_samples, labels)):
                        try:
                            # Extract label value
                            label_val = self._extract_label(label_item)
                            
                            # Combine data with label and language
                            combined_item = {**data_item}
                            combined_item['label'] = label_val
                            combined_item['language'] = lang
                            combined_item['source'] = 'train'
                            
                            all_data.append(combined_item)
                        except Exception as e:
                            logger.warning(f"Error processing {lang} item {i}: {e}")
                            continue
        
        # Load sampling data if available
        sampling_dir = self.data_path / "Sampling Data"
        if sampling_dir.exists():
            data_file = sampling_dir / "data.jsonl"
            label_file = sampling_dir / "label.jsonl"
            
            if data_file.exists() and label_file.exists():
                logger.info("📊 Loading sampling data...")
                
                data_samples = self._load_jsonl_file(data_file)
                labels = self._load_jsonl_file(label_file)
                
                for i, (data_item, label_item) in enumerate(zip(data_samples, labels)):
                    try:
                        label_val = self._extract_label(label_item)
                        
                        combined_item = {**data_item}
                        combined_item['label'] = label_val
                        combined_item['language'] = 'sampling'
                        combined_item['source'] = 'sampling'
                        
                        all_data.append(combined_item)
                    except Exception as e:
                        logger.warning(f"Error processing sampling item {i}: {e}")
                        continue
        
        # Convert to DataFrame
        self.train_data = pd.DataFrame(all_data)
        
        if len(self.train_data) > 0:
            # Log statistics
            logger.info(f"✅ Loaded {len(self.train_data)} training samples")
            
            if 'label' in self.train_data.columns:
                label_dist = self.train_data['label'].value_counts()
                logger.info(f"📊 Label distribution: {dict(label_dist)}")
                
                # Handle class imbalance
                self._handle_class_imbalance()
            
            if 'language' in self.train_data.columns:
                lang_dist = self.train_data['language'].value_counts()
                logger.info(f"🌍 Language distribution: {dict(lang_dist)}")
        else:
            logger.error("❌ No training data loaded!")
        
        return self.train_data
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data"""
        logger.info("📊 Loading SHROOM test data...")
        
        all_data = []
        test_dir = self.data_path / "Test"
        
        if test_dir.exists():
            test_files = list(test_dir.glob("*_test_data.jsonl"))
            logger.info(f"Found {len(test_files)} test files: {[f.name for f in test_files]}")
            
            for test_file in test_files:
                lang = test_file.stem.split('_')[0]
                data_samples = self._load_jsonl_file(test_file)
                
                for item in data_samples:
                    item['language'] = lang
                    item['source'] = 'test'
                    all_data.append(item)
        
        self.test_data = pd.DataFrame(all_data)
        logger.info(f"✅ Loaded {len(self.test_data)} test samples")
        
        if len(self.test_data) > 0 and 'language' in self.test_data.columns:
            lang_dist = self.test_data['language'].value_counts()
            logger.info(f"🌍 Test language distribution: {dict(lang_dist)}")
        
        return self.test_data
    
    def _load_jsonl_file(self, file_path: Path) -> List[Dict]:
        """Load JSONL file safely"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON decode error in {file_path} line {line_num}: {e}")
                            continue
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
        
        return data
    
    def _extract_label(self, label_item: Dict) -> int:
        """Extract label value from label item"""
        # SHROOM-CAP specific: check has_factual_mistakes first
        if 'has_factual_mistakes' in label_item:
            val = label_item['has_factual_mistakes']
            if isinstance(val, str):
                return 1 if val.lower() in ['y', 'yes', 'true', '1'] else 0
            return 1 if val else 0
        
        # Then check has_fluency_mistakes as backup
        if 'has_fluency_mistakes' in label_item:
            val = label_item['has_fluency_mistakes']
            if isinstance(val, str):
                return 1 if val.lower() in ['y', 'yes', 'true', '1'] else 0
            return 1 if val else 0
        
        # Try common label keys
        for key in ['label', 'hallucination', 'target', 'gold', 'class', 'value']:
            if key in label_item:
                val = label_item[key]
                if isinstance(val, (int, bool)):
                    return 1 if val else 0
                elif isinstance(val, str):
                    if val.lower() in ['true', '1', 'yes', 'y', 'hallucination']:
                        return 1
                    else:
                        return 0
        
        # If it's a simple value
        if isinstance(label_item, (int, bool)):
            return 1 if label_item else 0
        
        # Default to 0
        return 0
    
    def _handle_class_imbalance(self):
        """Handle severe class imbalance"""
        if 'label' not in self.train_data.columns:
            return
        
        unique_labels = set(self.train_data['label'].unique())
        
        if len(unique_labels) == 1:
            logger.warning("⚠️  Only one class found in training data!")
            only_label = list(unique_labels)[0]
            
            if only_label == 0:
                # Create synthetic positive samples
                logger.info("Creating synthetic positive samples for training...")
                synthetic_count = min(50, len(self.train_data) // 10)
                synthetic_samples = self.train_data.sample(n=synthetic_count).copy()
                synthetic_samples['label'] = 1
                synthetic_samples['source'] = 'synthetic'
                
                self.train_data = pd.concat([self.train_data, synthetic_samples], ignore_index=True)
                
                new_dist = self.train_data['label'].value_counts()
                logger.info(f"📊 Updated distribution: {dict(new_dist)}")
    
    def create_cv_folds(self, n_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create cross-validation folds with stratification"""
        logger.info(f"🔄 Creating {n_folds} CV folds...")
        
        if self.train_data is None or len(self.train_data) == 0:
            raise ValueError("No training data available")
        
        # Create stratification key
        if 'label' in self.train_data.columns and 'language' in self.train_data.columns:
            stratify_key = (
                self.train_data['label'].astype(str) + "_" + 
                self.train_data['language'].astype(str)
            )
        elif 'label' in self.train_data.columns:
            stratify_key = self.train_data['label']
        else:
            raise ValueError("No label column for stratification")
        
        # Create folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.train_data, stratify_key)):
            train_fold = self.train_data.iloc[train_idx].reset_index(drop=True)
            val_fold = self.train_data.iloc[val_idx].reset_index(drop=True)
            
            folds.append((train_fold, val_fold))
            
            logger.info(f"Fold {fold_idx + 1}: Train={len(train_fold)}, Val={len(val_fold)}")
        
        return folds
    
    def create_datasets(self, tokenizer_name: str, max_length: int = 512, n_folds: int = 5) -> Dict:
        """Create PyTorch datasets for training"""
        logger.info(f"🔧 Creating datasets with {tokenizer_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        datasets = {'tokenizer': tokenizer}
        
        # Create training folds if we have training data
        if self.train_data is not None and len(self.train_data) > 0:
            folds = self.create_cv_folds(n_folds)
            datasets['train_folds'] = []
            
            for fold_idx, (train_fold, val_fold) in enumerate(folds):
                train_dataset = SHROOMDataset(train_fold, tokenizer, max_length, is_training=True)
                val_dataset = SHROOMDataset(val_fold, tokenizer, max_length, is_training=True)
                
                datasets['train_folds'].append({
                    'train': train_dataset,
                    'val': val_dataset,
                    'fold': fold_idx
                })
        
        # Create test dataset if available
        if self.test_data is not None and len(self.test_data) > 0:
            datasets['test'] = SHROOMDataset(self.test_data, tokenizer, max_length, is_training=False)
        
        # Save tokenizer
        tokenizer_path = self.storage_path / f"tokenizer_{tokenizer_name.replace('/', '_')}"
        tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"💾 Tokenizer saved: {tokenizer_path}")
        
        return datasets
    
    def save_processed_data(self):
        """Save processed data for inspection"""
        if self.train_data is not None:
            train_path = self.storage_path / "processed_train_data.csv"
            self.train_data.to_csv(train_path, index=False)
            logger.info(f"💾 Training data saved: {train_path}")
        
        if self.test_data is not None:
            test_path = self.storage_path / "processed_test_data.csv"
            self.test_data.to_csv(test_path, index=False)
            logger.info(f"💾 Test data saved: {test_path}")

def main():
    """Test data processing"""
    processor = SHROOMDataProcessor()
    
    # Load data
    processor.load_training_data()
    processor.load_test_data()
    
    # Create datasets
    if processor.train_data is not None and len(processor.train_data) > 0:
        datasets = processor.create_datasets("distilbert-base-uncased", max_length=256, n_folds=3)
        logger.info(f"✅ Created datasets with {len(datasets.get('train_folds', []))} folds")
    
    # Save processed data
    processor.save_processed_data()
    
    logger.info("🎉 Data processing completed!")

if __name__ == "__main__":
    main()