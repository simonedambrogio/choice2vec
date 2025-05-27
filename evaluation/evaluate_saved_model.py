import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import os
import pickle
import argparse
from typing import Optional

from choice2vec_model import (
    Choice2Vec,
    Choice2VecTrainer,
    prepare_behavioral_data
)

# JAX will automatically use GPU if available
print("Available JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())


def load_model_state(model_path: str):
    """Load saved model state from pickle file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"📂 Loading model from: {model_path}")
    saved_data = pickle.load(open(model_path, "rb"))
    
    # Handle both old format (direct state) and new format (dict with params)
    if isinstance(saved_data, dict) and 'params' in saved_data:
        print(f"✅ Model loaded successfully! (New format)")
        return saved_data
    else:
        print(f"✅ Model loaded successfully! (Legacy format)")
        return {'params': saved_data}


def extract_representations(state, trainer, behavioral_features, environmental_features, batch_size=8):
    """Extract learned representations from the trained model."""
    behavioral_jax = jnp.array(behavioral_features)
    environmental_jax = jnp.array(environmental_features)
    
    all_outputs = []
    for i in range(0, len(behavioral_features), batch_size):
        behavioral_batch = behavioral_jax[i:i + batch_size]
        environmental_batch = environmental_jax[i:i + batch_size]
        
        outputs = state.apply_fn(
            {'params': state.params},
            behavioral_batch, environmental_batch,
            training=False
        )
        all_outputs.append(outputs)
    
    # Concatenate outputs
    contextualized_features = jnp.concatenate([out['contextualized_features'] for out in all_outputs], axis=0)
    quantized_indices = jnp.concatenate([out['quantized_indices'] for out in all_outputs], axis=0)
    
    return np.array(contextualized_features), np.array(quantized_indices)


def evaluate_state_classification(contextualized_features, states, test_size=0.3, verbose=True):
    """Evaluate how well the learned representations can classify psychological states."""
    
    # Pool features across sequence dimension (average)
    pooled_features = np.mean(contextualized_features, axis=1)  # [num_windows, embed_dim]
    pooled_states = np.mean(states, axis=1)  # [num_windows] - average engagement per window
    
    # Convert continuous states to binary classification (engaged vs disengaged)
    binary_states = (pooled_states > 0.5).astype(int)  # 1 = engaged, 0 = disengaged
    
    # Split data
    n_samples = len(pooled_features)
    n_train = int(n_samples * (1 - test_size))
    
    # Random split
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_test = pooled_features[train_idx], pooled_features[test_idx]
    y_train, y_test = binary_states[train_idx], binary_states[test_idx]
    
    # Train classifier
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"\n🎯 Window-Level Psychological State Classification Results:")
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"   Engaged samples: {np.sum(y_test)}/{len(y_test)} ({np.mean(y_test):.1%})")
        
        print(f"\n📊 Window-Level Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Disengaged', 'Engaged']))
    
    return accuracy, classifier, (X_train, X_test, y_train, y_test)


def evaluate_trial_level_classification(contextualized_features, states, test_size=0.3, verbose=True):
    """Evaluate psychological state classification at the individual trial level."""
    
    # Flatten to get individual trial representations
    trial_features = contextualized_features.reshape(-1, contextualized_features.shape[-1])  # [num_trials, embed_dim]
    trial_states = states.reshape(-1)  # [num_trials] - true state for each trial
    
    # Convert continuous states to binary classification (engaged vs disengaged)
    binary_states = (trial_states > 0.5).astype(int)  # 1 = engaged, 0 = disengaged
    
    # Split data
    n_samples = len(trial_features)
    n_train = int(n_samples * (1 - test_size))
    
    # Random split
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_test = trial_features[train_idx], trial_features[test_idx]
    y_train, y_test = binary_states[train_idx], binary_states[test_idx]
    
    # Train classifier
    classifier = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"\n🎯 Trial-Level Psychological State Classification Results:")
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(f"   Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
        print(f"   Total trials classified: {len(trial_features):,}")
        print(f"   Engaged samples: {np.sum(y_test):,}/{len(y_test):,} ({np.mean(y_test):.1%})")
        
        print(f"\n📊 Trial-Level Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Disengaged', 'Engaged']))
    
    return accuracy, classifier, (X_train, X_test, y_train, y_test)


def visualize_representations(contextualized_features, states, save_plots=True, plot_name="model_evaluation"):
    """Create visualizations of the learned representations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. PCA of learned representations (window-level)
    ax1 = axes[0, 0]
    ax1.set_title('Window-Level Representations (PCA)', fontsize=14, fontweight='bold')
    
    # Pool features and states
    pooled_features = np.mean(contextualized_features, axis=1)
    pooled_states = np.mean(states, axis=1)
    
    # Apply PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(pooled_features)
    
    scatter = ax1.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=pooled_states, cmap='RdYlBu', alpha=0.7, s=30)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=ax1, label='Engagement Level')
    
    # 2. PCA of trial-level representations
    ax2 = axes[0, 1]
    ax2.set_title('Trial-Level Representations (PCA)', fontsize=14, fontweight='bold')
    
    # Flatten to trial level
    trial_features = contextualized_features.reshape(-1, contextualized_features.shape[-1])
    trial_states = states.reshape(-1)
    
    # Sample for visualization (too many points otherwise)
    n_sample = min(2000, len(trial_features))
    sample_idx = np.random.choice(len(trial_features), n_sample, replace=False)
    
    trial_features_sample = trial_features[sample_idx]
    trial_states_sample = trial_states[sample_idx]
    
    # Apply PCA
    pca_trial = PCA(n_components=2)
    trial_features_pca = pca_trial.fit_transform(trial_features_sample)
    
    scatter2 = ax2.scatter(trial_features_pca[:, 0], trial_features_pca[:, 1], 
                          c=trial_states_sample, cmap='RdYlBu', alpha=0.6, s=10)
    ax2.set_xlabel(f'PC1 ({pca_trial.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca_trial.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter2, ax=ax2, label='Engagement Level')
    
    # 3. Engagement distribution over time
    ax3 = axes[1, 0]
    ax3.set_title('Engagement Over Time', fontsize=14, fontweight='bold')
    
    # Average engagement per window
    window_engagement = np.mean(states, axis=1)
    ax3.plot(window_engagement, alpha=0.7, linewidth=2)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax3.set_xlabel('Window Index')
    ax3.set_ylabel('Average Engagement')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature variance explained
    ax4 = axes[1, 1]
    ax4.set_title('PCA Variance Explained', fontsize=14, fontweight='bold')
    
    # Compute PCA on pooled features
    pca_full = PCA()
    pca_full.fit(pooled_features)
    
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax4.plot(range(1, min(21, len(cumvar)+1)), cumvar[:20], 'bo-', linewidth=2, markersize=4)
    ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% Variance')
    ax4.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% Variance')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Cumulative Variance Explained')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'{plot_name}_representations.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as '{plot_name}_representations.png'")
    
    plt.show()


def evaluate_saved_model(model_path: str, data_path: str = 'behavioral_data.csv', 
                        window_size: int = 100, stride: int = 20):
    """Complete evaluation pipeline for a saved model."""
    
    print("🔍 Choice2Vec: Evaluating Saved Model")
    print("=" * 50)
    
    # Load model
    saved_data = load_model_state(model_path)
    model_params = saved_data['params']
    
    # Ensure parameters are on the correct device (JAX will handle this automatically)
    # But we can explicitly place them if needed
    print(f"   Parameters loaded on device: {list(jax.tree.leaves(model_params))[0].device()}")
    
    # Load and prepare data
    if not os.path.exists(data_path):
        print(f"❌ Error: '{data_path}' not found!")
        print("   Please ensure the behavioral data file exists.")
        return None
    
    print(f"\n📊 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=window_size, stride=stride
    )
    
    print(f"   Loaded {len(df)} trials")
    print(f"   Created {len(behavioral_features)} windows")
    print(f"   Behavioral features: {behavioral_features.shape}")
    print(f"   Environmental features: {environmental_features.shape}")
    
    # Create trainer (needed for extraction)
    model = Choice2Vec(
        encoder_hidden_dims=(64, 128, 256),
        num_quantizer_groups=2,
        num_entries_per_group=128,
        num_transformer_layers=4,
        embed_dim=256,
        num_heads=4,
        dropout_rate=0.1,
        mask_prob=0.15
    )
    
    trainer = Choice2VecTrainer(
        model=model,
        learning_rate=1e-4,
        weight_decay=0.01,
        diversity_weight=0.1,
        contrastive_weight=1.0,
        use_cosine_loss=True
    )
    
    # Create a minimal state object for inference
    from flax.training import train_state
    import optax
    
    # Create a dummy state for inference (we only need params and apply_fn)
    class InferenceState:
        def __init__(self, params, apply_fn):
            self.params = params
            self.apply_fn = apply_fn
    
    inference_state = InferenceState(model_params, trainer.model.apply)
    
    # Extract representations
    print("\n🧠 Extracting learned representations...")
    contextualized_features, quantized_indices = extract_representations(
        inference_state, trainer, behavioral_features, environmental_features
    )
    
    print(f"   Extracted representations: {contextualized_features.shape}")
    
    # Evaluate window-level classification
    print("\n🎯 Evaluating window-level psychological state classification...")
    window_results = evaluate_state_classification(
        contextualized_features, states, test_size=0.3, verbose=True
    )
    
    # Evaluate trial-level classification
    print("\n🎯 Evaluating trial-level psychological state classification...")
    trial_results = evaluate_trial_level_classification(
        contextualized_features, states, test_size=0.3, verbose=True
    )
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    visualize_representations(
        contextualized_features, states, 
        save_plots=True, plot_name=f"{model_name}_evaluation"
    )
    
    # Summary
    window_accuracy = window_results[0]
    trial_accuracy = trial_results[0]
    
    print(f"\n🎉 Evaluation Complete!")
    print(f"=" * 40)
    print(f"✅ Window-level accuracy: {window_accuracy:.1%}")
    print(f"✅ Trial-level accuracy: {trial_accuracy:.1%}")
    print(f"✅ Total trials analyzed: {contextualized_features.shape[0] * contextualized_features.shape[1]:,}")
    print(f"✅ Model parameters: {sum(x.size for x in jax.tree.leaves(model_params)):,}")
    
    return {
        'saved_data': saved_data,
        'params': model_params,
        'data': (behavioral_features, environmental_features, states),
        'representations': (contextualized_features, quantized_indices),
        'window_results': window_results,
        'trial_results': trial_results
    }


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate saved Choice2Vec model')
    parser.add_argument('model_path', type=str, help='Path to saved model (.pkl file)')
    parser.add_argument('--data_path', type=str, default='behavioral_data.csv', 
                       help='Path to behavioral data CSV file')
    parser.add_argument('--window_size', type=int, default=100, 
                       help='Window size for data preparation')
    parser.add_argument('--stride', type=int, default=20, 
                       help='Stride for sliding windows')
    
    args = parser.parse_args()
    
    results = evaluate_saved_model(
        model_path=args.model_path,
        data_path=args.data_path,
        window_size=args.window_size,
        stride=args.stride
    )
    
    return results


if __name__ == "__main__":
    main() 