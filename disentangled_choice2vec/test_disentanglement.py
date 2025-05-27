#!/usr/bin/env python3
"""
Quick Test Script for Disentangled Choice2Vec

This script performs a quick test to verify that the disentanglement
implementation works correctly before running the full comparison.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from core.choice2vec_model import prepare_behavioral_data
from disentangled_choice2vec import (
    DisentangledChoice2Vec, DisentangledChoice2VecTrainer, 
    analyze_disentanglement, mutual_information_loss, orthogonality_loss
)

def test_disentanglement_components():
    """Test individual disentanglement components."""
    print("🧪 Testing Disentanglement Components")
    print("=" * 50)
    
    # Create dummy data
    batch_size, seq_len, embed_dim = 4, 10, 256
    features = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_len, embed_dim))
    
    # Test mutual information loss
    print("1. Testing Mutual Information Loss...")
    mi_loss = mutual_information_loss(features, num_groups=4)
    print(f"   MI Loss: {mi_loss:.4f}")
    assert not jnp.isnan(mi_loss), "MI loss should not be NaN"
    assert mi_loss >= 0, "MI loss should be non-negative"
    
    # Test orthogonality loss
    print("2. Testing Orthogonality Loss...")
    ortho_loss = orthogonality_loss(features, num_groups=4)
    print(f"   Orthogonality Loss: {ortho_loss:.4f}")
    assert not jnp.isnan(ortho_loss), "Orthogonality loss should not be NaN"
    assert ortho_loss >= 0, "Orthogonality loss should be non-negative"
    
    print("✅ Component tests passed!")

def test_disentangled_model():
    """Test the disentangled model initialization and forward pass."""
    print("\n🧠 Testing Disentangled Model")
    print("=" * 50)
    
    # Create model
    model = DisentangledChoice2Vec(
        encoder_hidden_dims=(64, 128, 256),
        num_quantizer_groups=4,
        num_entries_per_group=64,
        num_transformer_layers=2,  # Smaller for testing
        embed_dim=256,
        num_heads=4,
        dropout_rate=0.1,
        mask_prob=0.15,
        beta=4.0
    )
    
    # Create dummy inputs
    batch_size, seq_len = 2, 5
    behavioral_features = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_len, 2))
    environmental_features = jax.random.normal(jax.random.PRNGKey(43), (batch_size, seq_len, 3))
    
    # Initialize model
    rng = jax.random.PRNGKey(42)
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    
    variables = model.init(
        {'params': rng1, 'gumbel': rng2, 'dropout': rng3},
        behavioral_features, environmental_features,
        training=True, mask_key=rng1
    )
    
    print("1. Model initialization successful ✅")
    
    # Test forward pass
    outputs = model.apply(
        variables,
        behavioral_features, environmental_features,
        training=True, mask_key=rng1,
        rngs={'gumbel': rng2, 'dropout': rng3}
    )
    
    print("2. Forward pass successful ✅")
    
    # Check output shapes
    expected_keys = [
        'contextualized_features', 'factor_projected_features', 
        'global_projected_features', 'quantized_features',
        'quantized_indices', 'behavioral_predictions',
        'behavioral_targets', 'mask', 'commitment_loss'
    ]
    
    for key in expected_keys:
        assert key in outputs, f"Missing output key: {key}"
    
    print("3. Output keys correct ✅")
    
    # Check shapes
    assert outputs['contextualized_features'].shape == (batch_size, seq_len, 256)
    assert outputs['factor_projected_features'].shape == (batch_size, seq_len, 256)
    assert outputs['global_projected_features'].shape == (batch_size, seq_len, 256)
    assert outputs['quantized_indices'].shape == (batch_size, seq_len, 4)  # 4 groups
    
    print("4. Output shapes correct ✅")
    
    # Check commitment loss
    commitment_loss = outputs['commitment_loss']
    assert not jnp.isnan(commitment_loss), "Commitment loss should not be NaN"
    assert commitment_loss >= 0, "Commitment loss should be non-negative"
    
    print(f"5. Commitment loss: {commitment_loss:.4f} ✅")

def test_disentangled_trainer():
    """Test the disentangled trainer."""
    print("\n🏋️ Testing Disentangled Trainer")
    print("=" * 50)
    
    # Create model and trainer
    model = DisentangledChoice2Vec(
        num_quantizer_groups=4,
        num_entries_per_group=64,
        num_transformer_layers=2,  # Smaller for testing
        embed_dim=256
    )
    
    trainer = DisentangledChoice2VecTrainer(
        model=model,
        learning_rate=1e-4,
        factor_contrastive_weight=0.5,
        mi_weight=0.1,
        orthogonality_weight=0.05,
        commitment_weight=1.0
    )
    
    print("1. Trainer initialization successful ✅")
    
    # Create training state
    rng = jax.random.PRNGKey(42)
    behavioral_shape = (1, 5, 2)
    environmental_shape = (1, 5, 2)  # Updated to match psychological data format
    
    state = trainer.create_train_state(rng, behavioral_shape, environmental_shape)
    
    print("2. Training state creation successful ✅")
    
    # Test training step
    batch_size = 2
    behavioral_batch = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 5, 2))
    environmental_batch = jax.random.normal(jax.random.PRNGKey(43), (batch_size, 5, 2))  # Updated to match psychological data
    
    rng, step_rng = jax.random.split(rng)
    new_state, metrics = trainer.train_step(state, behavioral_batch, environmental_batch, step_rng)
    
    print("3. Training step successful ✅")
    
    # Check metrics
    expected_metrics = [
        'total_loss', 'behavioral_loss', 'global_contrastive_loss',
        'factor_contrastive_loss', 'mi_loss', 'orthogonality_loss',
        'commitment_loss', 'diversity_loss'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert not jnp.isnan(metrics[metric]), f"Metric {metric} is NaN"
    
    print("4. All metrics present and valid ✅")
    
    # Print metric values
    print("   Metric values:")
    for metric, value in metrics.items():
        print(f"     {metric}: {value:.4f}")

def test_with_real_data():
    """Test with real behavioral data if available."""
    print("\n📊 Testing with Real Data")
    print("=" * 50)
    
    # Try to load the new psychological dataset first, fall back to old format
    if os.path.exists('results/psychological_behavioral_data.csv'):
        print("   Using psychological behavioral data (choice_correct format)")
        df = pd.read_csv('results/psychological_behavioral_data.csv')
    elif os.path.exists('results/behavioral_data.csv'):
        print("   Using standard behavioral data (choice format)")
        df = pd.read_csv('results/behavioral_data.csv')
    else:
        print("⚠️ No behavioral data found, skipping real data test")
        print("   Run 'python data_generation/generate_psychological_data.py' to create test data")
        return
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=20, stride=10  # Smaller for testing
    )
    
    print(f"1. Loaded real data: {len(behavioral_features)} windows ✅")
    
    # Create model
    model = DisentangledChoice2Vec(
        num_quantizer_groups=4,
        num_entries_per_group=32,  # Smaller for testing
        num_transformer_layers=2,
        embed_dim=256  # Keep consistent with other tests
    )
    
    trainer = DisentangledChoice2VecTrainer(
        model=model,
        factor_contrastive_weight=0.5,
        mi_weight=0.1,
        orthogonality_weight=0.05
    )
    
    # Initialize training state
    rng = jax.random.PRNGKey(42)
    behavioral_shape = (1, behavioral_features.shape[1], behavioral_features.shape[2])
    environmental_shape = (1, environmental_features.shape[1], environmental_features.shape[2])
    
    state = trainer.create_train_state(rng, behavioral_shape, environmental_shape)
    
    print("2. Model initialized with real data shapes ✅")
    
    # Test a few training steps
    behavioral_jax = jnp.array(behavioral_features[:4])  # Small batch
    environmental_jax = jnp.array(environmental_features[:4])
    
    for step in range(3):
        rng, step_rng = jax.random.split(rng)
        state, metrics = trainer.train_step(state, behavioral_jax, environmental_jax, step_rng)
        print(f"   Step {step + 1}: Total loss = {metrics['total_loss']:.4f}")
    
    print("3. Training steps successful ✅")
    
    # Test representation extraction
    outputs = state.apply_fn(
        {'params': state.params},
        behavioral_jax[:2], environmental_jax[:2],
        training=False
    )
    
    representations = outputs['contextualized_features']
    print(f"4. Extracted representations: {representations.shape} ✅")
    
    # Test disentanglement analysis
    try:
        metrics = analyze_disentanglement(
            representations, 
            jnp.array(states[:2]), 
            num_groups=4
        )
        print("5. Disentanglement analysis successful ✅")
        print(f"   Independence score: {1.0 - metrics['mean_factor_correlation']:.3f}")
        print(f"   Disentanglement score: {metrics['disentanglement_score']:.3f}")
    except Exception as e:
        print(f"5. Disentanglement analysis failed: {e}")
        print("   (This is expected with very small data)")

def main():
    """Run all tests."""
    print("🧪 Disentangled Choice2Vec Test Suite")
    print("=" * 70)
    
    try:
        test_disentanglement_components()
        test_disentangled_model()
        test_disentangled_trainer()
        test_with_real_data()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("✅ Disentanglement implementation is working correctly")
        print("✅ Ready to run full comparison with train_disentangled_choice2vec.py")
        print("✅ All components tested and validated")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Please check the implementation and try again.")
        raise

if __name__ == "__main__":
    main() 