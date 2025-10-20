#!/usr/bin/env python3
"""
🔄 Unified Data Loader for All Datasets
========================================

This module handles loading data from ALL different dataset formats:
1. hallucination_dataset_100k.csv (100K samples)
2. SHROOM TrainSet V2 (JSONL with separate labels)
3. Augmented training data (merged JSONL)
4. LibreEval (prompt/completion CSV format)

All datasets are converted to a unified format:
{
    'text': str,           # The input text to classify
    'label': int,          # 0 = CORRECT, 1 = HALLUCINATED
    'source': str,         # Which dataset it came from
    'language': str,       # en, es, fr, it, etc.
    'metadata': dict       # Additional info (optional)
}
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass, asdict
import random


@dataclass
class UnifiedSample:
    """Unified format for all training samples"""
    text: str
    label: int  # 0 = CORRECT, 1 = HALLUCINATED
    source: str
    language: str
    metadata: Optional[Dict] = None
    
    def to_dict(self):
        return asdict(self)


class UnifiedDataLoader:
    """
    Load and unify data from all different dataset formats
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples: List[UnifiedSample] = []
        
    def load_all(self, max_samples: Optional[int] = None) -> List[UnifiedSample]:
        """
        Load ALL datasets and return unified samples
        
        Args:
            max_samples: Maximum number of samples to load (None = load all)
            
        Returns:
            List of UnifiedSample objects
        """
        print("=" * 80)
        print("🔄 UNIFIED DATA LOADER - Loading ALL Datasets")
        print("=" * 80)
        
        # 1. Load hallucination_dataset_100k.csv
        self._load_100k_dataset()
        
        # 2. Load SHROOM TrainSet V2
        self._load_shroom_trainset()
        
        # 3. Load augmented data
        self._load_augmented_data()
        
        # 4. Load LibreEval
        self._load_libreeval()
        
        # Shuffle samples
        random.shuffle(self.samples)
        
        # Limit if requested
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        # Print summary
        self._print_summary()
        
        return self.samples
    
    def _load_100k_dataset(self):
        """
        Load hallucination_dataset_100k.csv
        
        Format:
        id,category,difficulty,label,question,context,answer,hallucination_type,explanation,correct_answer
        """
        csv_path = self.data_dir / "hallucination_dataset_100k.csv"
        
        if not csv_path.exists():
            print(f"⚠️  Skipping: {csv_path} not found")
            return
        
        print(f"\n📂 Loading: {csv_path.name}")
        count = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract fields
                label_str = row.get('label', '').upper()
                question = row.get('question', '')
                context = row.get('context', '')
                answer = row.get('answer', '')
                
                # Combine into text
                text_parts = []
                if question:
                    text_parts.append(f"Question: {question}")
                if context:
                    text_parts.append(f"Context: {context}")
                if answer:
                    text_parts.append(f"Answer: {answer}")
                
                text = " ".join(text_parts)
                
                # Map label
                if label_str == 'CORRECT':
                    label = 0
                elif label_str == 'HALLUCINATED':
                    label = 1
                else:
                    print(f"⚠️  Unknown label: {label_str}, skipping")
                    continue
                
                # Create unified sample
                sample = UnifiedSample(
                    text=text,
                    label=label,
                    source="hallucination_dataset_100k",
                    language="en",  # Assume English
                    metadata={
                        'id': row.get('id'),
                        'category': row.get('category'),
                        'difficulty': row.get('difficulty'),
                        'hallucination_type': row.get('hallucination_type')
                    }
                )
                
                self.samples.append(sample)
                count += 1
        
        print(f"   ✅ Loaded {count:,} samples")
    
    def _load_shroom_trainset(self):
        """
        Load SHROOM TrainSet V2
        
        Format:
        - en_train2_data.jsonl: {index, title, abstract, question, output_text, ...}
        - en_train2_label.jsonl: {index, has_fluency_mistakes, has_factual_mistakes}
        """
        trainset_dir = self.data_dir / "TrainSet V2"
        
        if not trainset_dir.exists():
            print(f"⚠️  Skipping: {trainset_dir} not found")
            return
        
        print(f"\n📂 Loading: SHROOM TrainSet V2")
        
        for lang in ['en', 'es', 'fr', 'it', 'hi', 'bn', 'gu', 'ml', 'te']:
            data_file = trainset_dir / f"{lang}_train2_data.jsonl"
            label_file = trainset_dir / f"{lang}_train2_label.jsonl"
            
            if not data_file.exists() or not label_file.exists():
                continue
            
            # Load labels into dict
            labels_dict = {}
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    label_obj = json.loads(line)
                    index = label_obj['index']
                    # If has_factual_mistakes == 'y', it's a hallucination
                    has_mistakes = label_obj.get('has_factual_mistakes', 'n') == 'y'
                    labels_dict[index] = 1 if has_mistakes else 0
            
            # Load data
            count = 0
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data_obj = json.loads(line)
                    index = data_obj['index']
                    
                    if index not in labels_dict:
                        continue
                    
                    # Extract text fields
                    title = data_obj.get('title', '')
                    abstract = data_obj.get('abstract', '')
                    question = data_obj.get('question', '')
                    output_text = data_obj.get('output_text', '')
                    
                    # Combine into text
                    text_parts = []
                    if title:
                        text_parts.append(f"Title: {title}")
                    if abstract:
                        text_parts.append(f"Abstract: {abstract}")
                    if question:
                        text_parts.append(f"Question: {question}")
                    if output_text:
                        text_parts.append(f"Answer: {output_text}")
                    
                    text = " ".join(text_parts)
                    
                    # Create unified sample
                    sample = UnifiedSample(
                        text=text,
                        label=labels_dict[index],
                        source="shroom_trainset_v2",
                        language=lang,
                        metadata={
                            'index': index,
                            'model_id': data_obj.get('model_id')
                        }
                    )
                    
                    self.samples.append(sample)
                    count += 1
            
            print(f"   ✅ {lang}: {count} samples")
    
    def _load_augmented_data(self):
        """
        Load augmented training data (merged files)
        
        Format: Same as SHROOM TrainSet V2 (merged data + labels)
        """
        aug_dir = self.data_dir.parent / "augmented_training_data"
        
        if not aug_dir.exists():
            print(f"⚠️  Skipping: {aug_dir} not found")
            return
        
        print(f"\n📂 Loading: Augmented Training Data")
        
        for lang in ['en', 'es', 'fr', 'it']:
            data_file = aug_dir / f"{lang}_train_merged_data.jsonl"
            label_file = aug_dir / f"{lang}_train_merged_labels.jsonl"
            
            if not data_file.exists() or not label_file.exists():
                continue
            
            # Load labels
            labels_dict = {}
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    label_obj = json.loads(line)
                    index = label_obj['index']
                    has_mistakes = label_obj.get('has_factual_mistakes', 'n') == 'y'
                    labels_dict[index] = 1 if has_mistakes else 0
            
            # Load data
            count = 0
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data_obj = json.loads(line)
                    index = data_obj['index']
                    
                    if index not in labels_dict:
                        continue
                    
                    # Extract text
                    question = data_obj.get('question', '')
                    output_text = data_obj.get('output_text', '')
                    title = data_obj.get('title', '')
                    
                    text_parts = []
                    if title:
                        text_parts.append(f"Title: {title}")
                    if question:
                        text_parts.append(f"Question: {question}")
                    if output_text:
                        text_parts.append(f"Answer: {output_text}")
                    
                    text = " ".join(text_parts)
                    
                    sample = UnifiedSample(
                        text=text,
                        label=labels_dict[index],
                        source="augmented_data",
                        language=lang,
                        metadata={'index': index}
                    )
                    
                    self.samples.append(sample)
                    count += 1
            
            print(f"   ✅ {lang}: {count} samples")
    
    def _load_libreeval(self):
        """
        Load LibreEval dataset
        
        Format:
        prompt,completion
        "In this task, you will...",Label: not supported
        
        The completion contains various labels:
        - "Label: supported" / "Label: factual" = CORRECT (0)
        - "Label: not supported" / "Label: hallucinated" = HALLUCINATED (1)
        """
        libre_dir = self.data_dir / "LibreEval/combined_datasets_for_tuning/all_languages/prompt_completion_format"
        train_file = libre_dir / "train.csv"
        
        if not train_file.exists():
            print(f"⚠️  Skipping: {train_file} not found")
            return
        
        print(f"\n📂 Loading: LibreEval")
        count = 0
        skipped = 0
        
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompt = row.get('prompt', '')
                    completion = row.get('completion', '')
                    
                    if not prompt or not completion:
                        skipped += 1
                        continue
                    
                    # Extract label from completion
                    # Look for multiple label formats
                    completion_lower = completion.lower().strip()
                    
                    # CORRECT labels (0)
                    if any(phrase in completion_lower for phrase in [
                        'label: supported',
                        'label:supported',
                        'label: factual',
                        'label:factual',
                        'factual',
                        'supported'
                    ]):
                        label = 0  # CORRECT
                    # HALLUCINATED labels (1)
                    elif any(phrase in completion_lower for phrase in [
                        'label: not supported',
                        'label:not supported',
                        'label: hallucinated',
                        'label:hallucinated',
                        'not supported',
                        'hallucinated'
                    ]):
                        label = 1  # HALLUCINATED
                    else:
                        # Skip if we can't determine label
                        skipped += 1
                        continue
                    
                    # Use prompt as text (clean up extra whitespace)
                    text = ' '.join(prompt.split())
                    
                    # Try to detect language from prompt
                    language = 'en'  # Default
                    prompt_lower = prompt.lower()
                    
                    # Check for language indicators
                    if 'language:' in prompt_lower:
                        for lang_code in ['en', 'es', 'fr', 'pt', 'ja', 'ko', 'zh', 'de', 'it', 'ar']:
                            if f'language: {lang_code}' in prompt_lower or f'language:{lang_code}' in prompt_lower:
                                language = lang_code
                                break
                    
                    sample = UnifiedSample(
                        text=text,
                        label=label,
                        source="libreeval",
                        language=language,
                        metadata={'completion': completion[:100]}  # Store first 100 chars
                    )
                    
                    self.samples.append(sample)
                    count += 1
            
            print(f"   ✅ Loaded {count:,} samples")
            if skipped > 0:
                print(f"   ⚠️  Skipped {skipped:,} samples (unclear labels)")
        except Exception as e:
            print(f"   ❌ Error loading LibreEval: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_summary(self):
        """Print summary of loaded data"""
        print("\n" + "=" * 80)
        print("📊 DATA LOADING SUMMARY")
        print("=" * 80)
        
        # Total samples
        print(f"\n✅ Total samples loaded: {len(self.samples):,}")
        
        # By source
        print("\n📂 By Source:")
        sources = {}
        for sample in self.samples:
            sources[sample.source] = sources.get(sample.source, 0) + 1
        for source, count in sorted(sources.items()):
            print(f"   {source}: {count:,}")
        
        # By language
        print("\n🌍 By Language:")
        languages = {}
        for sample in self.samples:
            languages[sample.language] = languages.get(sample.language, 0) + 1
        for lang, count in sorted(languages.items()):
            print(f"   {lang}: {count:,}")
        
        # By label
        print("\n🏷️  By Label:")
        labels = {0: 0, 1: 0}
        for sample in self.samples:
            labels[sample.label] += 1
        print(f"   CORRECT (0): {labels[0]:,} ({labels[0]/len(self.samples)*100:.1f}%)")
        print(f"   HALLUCINATED (1): {labels[1]:,} ({labels[1]/len(self.samples)*100:.1f}%)")
        
        print("\n" + "=" * 80)
    
    def get_texts_and_labels(self):
        """Return lists of texts and labels for model training"""
        texts = [s.text for s in self.samples]
        labels = [s.label for s in self.samples]
        return texts, labels
    
    def save_unified_format(self, output_file: str):
        """Save all samples in unified JSONL format"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"\n💾 Saved {len(self.samples):,} samples to: {output_path}")


def main():
    """Test the unified data loader"""
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "SHROOM2025/SHROOM/SHROOM_DATA"
    
    # Load all data
    loader = UnifiedDataLoader(data_dir)
    samples = loader.load_all()
    
    # Show some examples
    print("\n" + "=" * 80)
    print("📋 SAMPLE EXAMPLES (First 3)")
    print("=" * 80)
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n{i}. Source: {sample.source} | Language: {sample.language} | Label: {sample.label}")
        print(f"   Text: {sample.text[:200]}...")
    
    # Save unified format
    loader.save_unified_format("unified_training_data.jsonl")


if __name__ == "__main__":
    main()
