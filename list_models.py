#!/usr/bin/env python3
"""
Utility script to list all saved Choice2Vec models in the current directory.
"""

import os
import glob
from datetime import datetime
import pickle

def list_saved_models():
    """List all saved Choice2Vec model files."""
    
    # Find all .pkl files that match the naming pattern
    model_files = glob.glob("choice2vec_model_*.pkl")
    
    if not model_files:
        print("❌ No saved Choice2Vec models found in current directory.")
        print("   Run 'python train_choice2vec.py' to train and save a model first.")
        return
    
    print("📂 Found Choice2Vec Models:")
    print("=" * 60)
    
    models_info = []
    
    for model_file in sorted(model_files):
        try:
            # Get file info
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
            
            # Try to load model to get parameter count
            try:
                with open(model_file, 'rb') as f:
                    state = pickle.load(f)
                    param_count = sum(x.size for x in state.params.values() if hasattr(x, 'size'))
                    param_count_str = f"{param_count:,}"
            except:
                param_count_str = "Unknown"
            
            models_info.append({
                'file': model_file,
                'size_mb': file_size,
                'modified': mod_time,
                'params': param_count_str
            })
            
        except Exception as e:
            print(f"⚠️  Error reading {model_file}: {e}")
    
    # Display models
    for i, info in enumerate(models_info, 1):
        print(f"{i}. {info['file']}")
        print(f"   📊 Size: {info['size_mb']:.1f} MB")
        print(f"   📅 Modified: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🧠 Parameters: {info['params']}")
        print()
    
    print(f"💡 To evaluate a model, run:")
    print(f"   python evaluate_saved_model.py <model_filename>")
    print()
    print(f"💡 Example:")
    if models_info:
        print(f"   python evaluate_saved_model.py {models_info[-1]['file']}")

if __name__ == "__main__":
    list_saved_models() 