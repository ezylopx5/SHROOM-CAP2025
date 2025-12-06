#!/usr/bin/env python3
"""
Cross-Platform Checkpoint System for SHROOM-CAP
Seamless migration between Lightning AI and Google Colab
"""

import os
import json
import pickle
import torch
import logging
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import signal
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckpointSystem:
    """Manages checkpoints and migration between platforms"""
    
    def __init__(self, experiment_name: str = "shroom_ensemble"):
        self.experiment_name = experiment_name
        self.platform = self._detect_platform()
        self.base_path = self._get_base_path()
        self.checkpoint_dir = self.base_path / "checkpoints" / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Platform-specific settings
        self.time_limit = self._get_time_limit()
        self.start_time = datetime.now()
        self.should_stop = False
        
        # Set up graceful shutdown handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"🔧 Platform: {self.platform}")
        logger.info(f"📁 Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"⏰ Time limit: {self.time_limit}")
    
    def _detect_platform(self) -> str:
        """Detect current execution platform"""
        if os.path.exists("/teamspace/studios/this_studio"):
            return "lightning_ai"
        elif os.path.exists("/content"):
            return "colab"
        else:
            return "local"
    
    def _get_base_path(self) -> Path:
        """Get base storage path for platform"""
        if self.platform == "lightning_ai":
            return Path("/teamspace/studios/this_studio")
        elif self.platform == "colab":
            return Path("/content/drive/MyDrive/SHROOM_CAP")
        else:
            return Path(".")
    
    def _get_time_limit(self) -> Optional[timedelta]:
        """Get time limit for current platform"""
        if self.platform == "lightning_ai":
            # Conservative: stop 15 minutes before credit exhaustion
            return timedelta(hours=3, minutes=15)  # A100 limit
        elif self.platform == "colab":
            # Colab session limit with buffer
            return timedelta(hours=11, minutes=30)
        else:
            return None  # No limit for local
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logger.info("🛑 Interrupt signal received, preparing for graceful shutdown...")
        self.should_stop = True
    
    def check_time_limit(self) -> bool:
        """Check if approaching time limit"""
        if not self.time_limit:
            return False
        
        elapsed = datetime.now() - self.start_time
        remaining = self.time_limit - elapsed
        
        if remaining.total_seconds() < 600:  # 10 minutes warning
            logger.warning(f"⏰ Time limit approaching! {remaining} remaining")
            return True
        
        if remaining.total_seconds() <= 0:
            logger.error("⏰ Time limit exceeded!")
            self.should_stop = True
            return True
        
        return False
    
    def save_training_state(self, state: Dict[str, Any]) -> Path:
        """Save complete training state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced state with metadata
        enhanced_state = {
            **state,
            'platform': self.platform,
            'timestamp': timestamp,
            'save_time': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
            'elapsed_time': str(datetime.now() - self.start_time)
        }
        
        # Save timestamped version
        state_file = self.checkpoint_dir / f"training_state_{timestamp}.pkl"
        with open(state_file, 'wb') as f:
            pickle.dump(enhanced_state, f)
        
        # Save as latest
        latest_file = self.checkpoint_dir / "latest_training_state.pkl"
        with open(latest_file, 'wb') as f:
            pickle.dump(enhanced_state, f)
        
        # Also save as JSON for inspection
        json_file = self.checkpoint_dir / f"training_state_{timestamp}.json"
        json_safe_state = self._make_json_safe(enhanced_state)
        with open(json_file, 'w') as f:
            json.dump(json_safe_state, f, indent=2)
        
        logger.info(f"💾 Training state saved: {state_file}")
        return state_file
    
    def load_training_state(self, checkpoint_file: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load training state"""
        if checkpoint_file is None:
            checkpoint_file = self.checkpoint_dir / "latest_training_state.pkl"
        
        if not checkpoint_file.exists():
            logger.info("📭 No previous training state found")
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            logger.info(f"📂 Loaded training state from: {checkpoint_file}")
            logger.info(f"   Platform: {state.get('platform', 'unknown')}")
            logger.info(f"   Save time: {state.get('save_time', 'unknown')}")
            logger.info(f"   Current fold: {state.get('current_fold', 'unknown')}")
            logger.info(f"   Current epoch: {state.get('current_epoch', 'unknown')}")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Error loading training state: {e}")
            return None
    
    def save_model_checkpoint(self, model, tokenizer, optimizer, scheduler, 
                            fold: int, epoch: int, step: int, metrics: Dict) -> Path:
        """Save model checkpoint with all components"""
        checkpoint_name = f"fold_{fold}_epoch_{epoch}_step_{step}"
        model_dir = self.checkpoint_dir / "models" / checkpoint_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model and tokenizer
            model.save_pretrained(model_dir / "model")
            tokenizer.save_pretrained(model_dir / "tokenizer")
            
            # Save training components
            training_state = {
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'fold': fold,
                'epoch': epoch,
                'step': step,
                'metrics': metrics,
                'platform': self.platform,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(training_state, model_dir / "training_components.pt")
            
            # Save metadata
            metadata = {
                'checkpoint_name': checkpoint_name,
                'fold': fold,
                'epoch': epoch,
                'step': step,
                'metrics': metrics,
                'platform': self.platform,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"🎯 Model checkpoint saved: {checkpoint_name}")
            return model_dir
            
        except Exception as e:
            logger.error(f"❌ Error saving model checkpoint: {e}")
            raise
    
    def load_model_checkpoint(self, checkpoint_dir: Path) -> Dict[str, Any]:
        """Load model checkpoint"""
        try:
            # Load metadata
            metadata_file = checkpoint_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Load training components
            training_file = checkpoint_dir / "training_components.pt"
            if training_file.exists():
                training_components = torch.load(training_file, map_location='cpu')
            else:
                training_components = {}
            
            return {
                'model_dir': checkpoint_dir / "model",
                'tokenizer_dir': checkpoint_dir / "tokenizer",
                'training_components': training_components,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint from {checkpoint_dir}: {e}")
            raise
    
    def create_migration_package(self, include_data: bool = False) -> Path:
        """Create migration package for platform transfer"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"shroom_migration_{self.platform}_to_{'colab' if self.platform == 'lightning_ai' else 'lightning_ai'}_{timestamp}.zip"
        package_path = self.base_path / package_name
        
        logger.info(f"📦 Creating migration package: {package_name}")
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all checkpoints
            self._add_directory_to_zip(zipf, self.checkpoint_dir, "checkpoints")
            
            # Add scripts
            script_files = [
                "shroom_data_processor.py",
                "checkpoint_system.py",
                "shroom_trainer.py",
                "shroom_predictor.py"
            ]
            
            for script in script_files:
                script_path = self.base_path / script
                if script_path.exists():
                    zipf.write(script_path, f"scripts/{script}")
            
            # Add processed data if requested and not too large
            if include_data:
                data_dir = self.base_path / "shroom_workspace"
                if data_dir.exists():
                    self._add_directory_to_zip(zipf, data_dir, "workspace", max_files=100)
            
            # Add migration instructions
            instructions = self._create_migration_instructions()
            zipf.writestr("MIGRATION_INSTRUCTIONS.md", instructions)
        
        file_size_mb = os.path.getsize(package_path) / (1024 * 1024)
        logger.info(f"📦 Migration package created: {package_path}")
        logger.info(f"📊 Package size: {file_size_mb:.1f} MB")
        
        return package_path
    
    def _add_directory_to_zip(self, zipf: zipfile.ZipFile, source_dir: Path, 
                            archive_prefix: str, max_files: int = 1000):
        """Add directory to zip file with limits"""
        file_count = 0
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file_count >= max_files:
                    logger.warning(f"⚠️  Reached file limit ({max_files}) for {source_dir}")
                    return
                
                file_path = Path(root) / file
                archive_path = archive_prefix + "/" + str(file_path.relative_to(source_dir))
                zipf.write(file_path, archive_path)
                file_count += 1
    
    def _create_migration_instructions(self) -> str:
        """Create migration instructions"""
        target_platform = "Google Colab" if self.platform == "lightning_ai" else "Lightning AI"
        
        instructions = f"""
# SHROOM-CAP Migration Instructions

## Migration from {self.platform.replace('_', ' ').title()} to {target_platform}

### Current Status:
- Source platform: {self.platform}
- Experiment: {self.experiment_name}
- Migration time: {datetime.now().isoformat()}
- Elapsed training time: {datetime.now() - self.start_time}

### Steps to Resume Training:

#### For Google Colab:
1. Mount Google Drive and create working directory:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   import os
   os.makedirs('/content/drive/MyDrive/SHROOM_CAP', exist_ok=True)
   os.chdir('/content/drive/MyDrive/SHROOM_CAP')
   ```

2. Upload this migration package to your Google Drive

3. Extract the package:
   ```bash
   !unzip shroom_migration_*.zip
   ```

4. Install dependencies:
   ```bash
   !pip install transformers>=4.35.0 torch torchvision torchaudio
   !pip install scikit-learn pandas numpy accelerate peft bitsandbytes
   ```

5. Resume training:
   ```bash
   python scripts/shroom_trainer.py --resume
   ```

#### For Lightning AI:
1. Upload this package to your Lightning AI studio
2. Extract in terminal:
   ```bash
   unzip shroom_migration_*.zip
   ```
3. Resume training:
   ```bash
   python scripts/shroom_trainer.py --resume
   ```

### What's Included:
- All model checkpoints
- Training state and progress
- Configuration files
- Python scripts
- Processed data (if available)

### Notes:
- Training will automatically detect the platform and adjust paths
- All progress and metrics will be preserved
- Time limits will be adjusted for the new platform

Happy training! 🚀
        """
        
        return instructions.strip()
    
    def _make_json_safe(self, obj: Any) -> Any:
        """Convert object to JSON-safe format"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return f"<{type(obj).__name__}>"
        elif hasattr(obj, '__dict__'):
            return f"<{type(obj).__name__}>"
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def get_resume_info(self) -> Dict[str, Any]:
        """Get information about resumable training"""
        info = {
            'platform': self.platform,
            'base_path': str(self.base_path),
            'checkpoint_dir': str(self.checkpoint_dir),
            'has_checkpoints': False,
            'training_progress': {},
            'available_models': [],
            'can_resume': False
        }
        
        # Check for training state
        latest_state = self.checkpoint_dir / "latest_training_state.pkl"
        if latest_state.exists():
            info['has_checkpoints'] = True
            
            state = self.load_training_state()
            if state:
                info['training_progress'] = {
                    'current_fold': state.get('current_fold', 0),
                    'current_epoch': state.get('current_epoch', 0),
                    'completed_folds': state.get('completed_folds', []),
                    'total_steps': state.get('total_steps', 0),
                    'best_scores': state.get('best_scores', {}),
                    'elapsed_time': state.get('elapsed_time', 'unknown')
                }
                info['can_resume'] = True
        
        # Check for model checkpoints
        models_dir = self.checkpoint_dir / "models"
        if models_dir.exists():
            info['available_models'] = [d.name for d in models_dir.iterdir() if d.is_dir()]
        
        return info
    
    def cleanup_old_checkpoints(self, keep_recent: int = 5):
        """Clean up old checkpoints to save space"""
        models_dir = self.checkpoint_dir / "models"
        if not models_dir.exists():
            return
        
        # Get all model checkpoints sorted by creation time
        checkpoints = [(d, d.stat().st_mtime) for d in models_dir.iterdir() if d.is_dir()]
        checkpoints.sort(key=lambda x: x[1], reverse=True)  # Newest first
        
        if len(checkpoints) > keep_recent:
            logger.info(f"🧹 Cleaning up old checkpoints, keeping {keep_recent} most recent...")
            
            for checkpoint_dir, _ in checkpoints[keep_recent:]:
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"🗑️  Removed old checkpoint: {checkpoint_dir.name}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not remove {checkpoint_dir}: {e}")

def main():
    """Test checkpoint system"""
    checkpoint_system = CheckpointSystem("test_experiment")
    
    # Test basic functionality
    logger.info("🧪 Testing checkpoint system...")
    
    # Show resume info
    resume_info = checkpoint_system.get_resume_info()
    logger.info("📋 Resume info:")
    for key, value in resume_info.items():
        logger.info(f"   {key}: {value}")
    
    # Test saving a dummy state
    dummy_state = {
        'current_fold': 0,
        'current_epoch': 5,
        'completed_folds': [],
        'total_steps': 1000,
        'best_scores': {'f1': 0.85, 'accuracy': 0.82}
    }
    
    state_file = checkpoint_system.save_training_state(dummy_state)
    logger.info(f"✅ Test state saved: {state_file}")
    
    # Test loading
    loaded_state = checkpoint_system.load_training_state()
    if loaded_state:
        logger.info("✅ Test state loaded successfully")
    
    logger.info("🎉 Checkpoint system test completed!")

if __name__ == "__main__":
    main()