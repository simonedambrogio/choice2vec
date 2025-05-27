#!/usr/bin/env python3
"""
Test script to verify the reorganized Choice2Vec codebase works correctly.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        from core.choice2vec_model import Choice2Vec, Choice2VecTrainer, prepare_behavioral_data
        print("✅ Core imports working")
        
        # Test disentangled imports
        from disentangled_choice2vec.disentangled_choice2vec import DisentangledChoice2Vec, DisentangledChoice2VecTrainer
        print("✅ Disentangled imports working")
        
        # Test standard imports
        from standard_choice2vec.train_choice2vec import train_choice2vec
        print("✅ Standard Choice2Vec imports working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_generation():
    """Test that data generation works."""
    print("\n📊 Testing data generation...")
    
    try:
        # Test basic data generation
        from data_generation.generate_data import BehavioralDataGenerator
        
        # Create small test dataset
        generator = BehavioralDataGenerator(
            n_images=3,
            n_subtasks=2,
            trials_per_subtask=10,  # Very small for testing
            random_seed=42
        )
        
        df = generator.generate_data()
        print(f"✅ Generated {len(df)} trials")
        
        # Test psychological data generation
        from data_generation.generate_psychological_data import PsychologicalDataGenerator
        
        psych_generator = PsychologicalDataGenerator(
            n_images=3,
            n_subtasks=2,
            trials_per_subtask=10,  # Very small for testing
            random_seed=42
        )
        
        psych_df = psych_generator.generate_data()
        print(f"✅ Generated {len(psych_df)} psychological trials")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_model_creation():
    """Test that models can be created."""
    print("\n🧠 Testing model creation...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Test standard model
        from core.choice2vec_model import Choice2Vec
        
        model = Choice2Vec(
            encoder_hidden_dims=(32, 64),  # Smaller for testing
            num_quantizer_groups=2,
            num_entries_per_group=32,
            num_transformer_layers=2,
            embed_dim=64,
            num_heads=2,
            dropout_rate=0.1,
            mask_prob=0.15
        )
        print("✅ Standard Choice2Vec model created")
        
        # Test disentangled model
        from disentangled_choice2vec.disentangled_choice2vec import DisentangledChoice2Vec
        
        disentangled_model = DisentangledChoice2Vec(
            encoder_hidden_dims=(32, 64),  # Smaller for testing
            num_quantizer_groups=2,
            num_entries_per_group=32,
            num_transformer_layers=2,
            embed_dim=64,
            num_heads=2,
            dropout_rate=0.1,
            mask_prob=0.15
        )
        print("✅ Disentangled Choice2Vec model created")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_folder_structure():
    """Test that the folder structure is correct."""
    print("\n📁 Testing folder structure...")
    
    expected_folders = [
        'core',
        'data_generation', 
        'standard_choice2vec',
        'disentangled_choice2vec',
        'checkpoints',
        'evaluation',
        'utils',
        'docs',
        'results'
    ]
    
    missing_folders = []
    for folder in expected_folders:
        if not os.path.exists(folder):
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"❌ Missing folders: {missing_folders}")
        return False
    else:
        print("✅ All expected folders present")
        
    # Check for key files
    key_files = [
        'core/choice2vec_model.py',
        'data_generation/generate_data.py',
        'standard_choice2vec/train_choice2vec.py',
        'disentangled_choice2vec/disentangled_choice2vec.py',
        'README.md'
    ]
    
    missing_files = []
    for file_path in key_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing key files: {missing_files}")
        return False
    else:
        print("✅ All key files present")
        
    return True

def main():
    """Run all tests."""
    print("🚀 Choice2Vec Organization Test Suite")
    print("=" * 60)
    
    # Set JAX to CPU mode for testing
    os.environ['JAX_PLATFORMS'] = 'cpu'
    
    tests = [
        test_folder_structure,
        test_imports,
        test_data_generation,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} crashed: {e}")
    
    print(f"\n🎉 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The reorganized codebase is working correctly.")
        print("\n📋 ORGANIZATION SUMMARY:")
        print("✅ Core models in core/")
        print("✅ Data generation in data_generation/")
        print("✅ Standard Choice2Vec in standard_choice2vec/")
        print("✅ Disentangled Choice2Vec in disentangled_choice2vec/")
        print("✅ Checkpoints and long-term training in checkpoints/")
        print("✅ Evaluation tools in evaluation/")
        print("✅ Documentation in docs/")
        print("✅ Results and data in results/")
        print("\n🎯 Ready for development and experimentation!")
    else:
        print("❌ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main() 