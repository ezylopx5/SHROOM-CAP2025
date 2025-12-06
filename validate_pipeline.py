#!/usr/bin/env python3
"""
Pre-flight validation script for SHROOM-CAP pipeline
Tests critical components before Lightning AI deployment
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data_structure():
    """Validate SHROOM data structure"""
    logger.info("🔍 Validating SHROOM data structure...")
    
    data_dir = Path("SHROOM_DATA")
    if not data_dir.exists():
        logger.error("❌ SHROOM_DATA directory not found!")
        return False
    
    # Check training data
    train_dir = data_dir / "TrainSet V1"
    if not train_dir.exists():
        logger.error("❌ TrainSet V1 directory not found!")
        return False
    
    # Check for data files
    data_files = list(train_dir.glob("*_train_data.jsonl"))
    label_files = list(train_dir.glob("*_train_label.jsonl"))
    
    logger.info(f"✅ Found {len(data_files)} data files")
    logger.info(f"✅ Found {len(label_files)} label files")
    
    if len(data_files) == 0:
        logger.error("❌ No training data files found!")
        return False
    
    if len(label_files) == 0:
        logger.error("❌ No training label files found!")
        return False
    
    # Check test data
    test_dir = data_dir / "Test"
    if test_dir.exists():
        test_files = list(test_dir.glob("*_test_data.jsonl"))
        logger.info(f"✅ Found {len(test_files)} test files")
    else:
        logger.warning("⚠️  No Test directory found")
    
    # Validate file pairs
    languages = set()
    for data_file in data_files:
        lang = data_file.stem.split('_')[0]
        languages.add(lang)
        
        label_file = train_dir / f"{lang}_train_label.jsonl"
        if not label_file.exists():
            logger.error(f"❌ Missing label file for {lang}")
            return False
    
    logger.info(f"✅ Found data for languages: {sorted(languages)}")
    
    # Test loading a sample
    try:
        sample_data_file = data_files[0]
        with open(sample_data_file, 'r') as f:
            sample_line = f.readline().strip()
            sample_data = json.loads(sample_line)
            
        logger.info("✅ Sample data structure:")
        logger.info(f"   Keys: {list(sample_data.keys())}")
        
        # Check essential fields
        essential_fields = ['question', 'output_text']
        for field in essential_fields:
            if field not in sample_data:
                logger.error(f"❌ Missing essential field: {field}")
                return False
        
        # Test label loading
        lang = sample_data_file.stem.split('_')[0]
        label_file = train_dir / f"{lang}_train_label.jsonl"
        with open(label_file, 'r') as f:
            sample_label_line = f.readline().strip()
            sample_label = json.loads(sample_label_line)
            
        logger.info(f"✅ Sample label structure: {sample_label}")
        
    except Exception as e:
        logger.error(f"❌ Error reading sample data: {e}")
        return False
    
    return True

def validate_scripts():
    """Validate all Python scripts compile correctly"""
    logger.info("🔍 Validating Python scripts...")
    
    scripts = [
        'train_xlm.py',
        'unified_data_loader.py',
        'generate_final_submission.py',
        'start_training_full_dataset.py'
    ]
    
    for script in scripts:
        if not Path(script).exists():
            logger.error(f"❌ Missing script: {script}")
            return False
        
        # Test compilation
        try:
            import py_compile
            py_compile.compile(script, doraise=True)
            logger.info(f"✅ {script} - syntax OK")
        except py_compile.PyCompileError as e:
            logger.error(f"❌ {script} - syntax error: {e}")
            return False
    
    return True

def validate_imports():
    """Test key imports work"""
    logger.info("🔍 Testing critical imports...")
    
    try:
        # Test data loader
        sys.path.insert(0, '.')
        from unified_data_loader import UnifiedDataLoader
        logger.info("✅ UnifiedDataLoader import OK")
        
        # Test train_xlm can be imported
        import train_xlm
        logger.info("✅ train_xlm import OK")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def validate_notebook():
    """Validate Colab notebook JSON"""
    logger.info("🔍 Validating Colab notebook...")
    
    notebook_file = Path("SHROOM_Colab_Migration.ipynb")
    if not notebook_file.exists():
        logger.error("❌ Colab notebook not found!")
        return False
    
    try:
        with open(notebook_file, 'r') as f:
            notebook_data = json.load(f)
        
        # Check basic structure
        if 'cells' not in notebook_data:
            logger.error("❌ Invalid notebook structure - no cells")
            return False
        
        logger.info(f"✅ Notebook has {len(notebook_data['cells'])} cells")
        
        # Check for GPU setting
        metadata = notebook_data.get('metadata', {})
        if 'accelerator' in metadata and metadata['accelerator'] == 'GPU':
            logger.info("✅ Notebook configured for GPU")
        else:
            logger.warning("⚠️  Notebook not configured for GPU")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid notebook JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error reading notebook: {e}")
        return False

def run_validation():
    """Run complete validation"""
    logger.info("🍄⚡ SHROOM-CAP Pipeline Validation")
    logger.info("="*50)
    
    all_valid = True
    
    # Validate data structure
    if not validate_data_structure():
        all_valid = False
    
    # Validate scripts
    if not validate_scripts():
        all_valid = False
    
    # Validate imports
    if not validate_imports():
        all_valid = False
    
    logger.info("="*50)
    
    if all_valid:
        logger.info("🎉 ALL VALIDATIONS PASSED!")
        logger.info("✅ Pipeline is ready for training")
        logger.info("")
        logger.info("📋 Next steps:")
        logger.info("1. Run: python start_training_full_dataset.py")
        logger.info("   OR")
        logger.info("   python train_xlm.py --model_name xlm-roberta-large --data_dir SHROOM_DATA --output_dir outputs")
        logger.info("")
        logger.info("2. After training, generate submissions:")
        logger.info("   python generate_final_submission.py --model_path <model_path> --data_dir SHROOM_DATA --output_dir submissions")
        logger.info("")
        logger.info("🚀 Good luck with your training!")
        return True
    else:
        logger.error("❌ VALIDATION FAILED!")
        logger.error("Please fix the issues above before training")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)