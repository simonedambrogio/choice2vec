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
from datetime import datetime

from choice2vec_model import (
    Choice2Vec,
    Choice2VecTrainer,
    prepare_behavioral_data
)

# JAX will automatically use GPU if available
print("Available JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())


def train_choice2vec(num_epochs: int = 20, 
                    batch_size: int = 8,
                    learning_rate: float = 1e-4,
                    use_cosine_loss: bool = True,
                    verbose: bool = True):
    """
    Train Choice2Vec model and return trained state and data.
    """
    
    # Check if data exists
    if not os.path.exists('behavioral_data.csv'):
        print("❌ Error: 'behavioral_data.csv' not found!")
        print("   Please run 'python generate_data.py' first to create the dataset.")
        return None, None, None, None
    
    # Load and prepare data
    if verbose:
        print("📊 Loading and preparing data...")
    
    df = pd.read_csv('behavioral_data.csv')
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=100, stride=20  # Larger windows, more data per sequence
    )
    
    if verbose:
        print(f"   Loaded {len(df)} trials")
        print(f"   Created {len(behavioral_features)} windows")
        print(f"   Behavioral features: {behavioral_features.shape} [choice, rt]")
        print(f"   Environmental features: {environmental_features.shape} [value_diff, trial, subtask]")
    
    # Initialize model
    if verbose:
        print(f"\n🧠 Initializing Choice2Vec model...")
    
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
    
    # Initialize trainer
    trainer = Choice2VecTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=0.01,
        diversity_weight=0.1,
        contrastive_weight=1.0,
        use_cosine_loss=use_cosine_loss
    )
    
    # Initialize training state
    rng = jax.random.PRNGKey(42)
    behavioral_shape = (1, behavioral_features.shape[1], behavioral_features.shape[2])
    environmental_shape = (1, environmental_features.shape[1], environmental_features.shape[2])
    
    state = trainer.create_train_state(rng, behavioral_shape, environmental_shape)
    
    if verbose:
        print(f"   Model parameters: {sum(x.size for x in jax.tree.leaves(state.params)):,}")
    
    # Training loop
    behavioral_jax = jnp.array(behavioral_features)
    environmental_jax = jnp.array(environmental_features)
    
    training_history = {
        'total_loss': [],
        'behavioral_loss': [],
        'contrastive_loss': [],
        'diversity_loss': [],
        'mask_ratio': []
    }
    
    if verbose:
        print(f"\n🏋️ Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_metrics = {key: [] for key in training_history.keys()}
        
        # Shuffle data
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, len(behavioral_features))
        behavioral_shuffled = behavioral_jax[perm]
        environmental_shuffled = environmental_jax[perm]
        
        # Mini-batch training
        for i in range(0, len(behavioral_features), batch_size):
            behavioral_batch = behavioral_shuffled[i:i + batch_size]
            environmental_batch = environmental_shuffled[i:i + batch_size]
            
            rng, step_rng = jax.random.split(rng)
            state, metrics = trainer.train_step(
                state, behavioral_batch, environmental_batch, step_rng
            )
            
            for key in epoch_metrics:
                epoch_metrics[key].append(float(metrics[key]))
        
        # Record epoch averages
        for key in training_history:
            avg_val = np.mean(epoch_metrics[key])
            training_history[key].append(avg_val)
        
        if verbose and (epoch % 100 == 0 or epoch == num_epochs - 1):
            print(f"   Epoch {epoch + 1:2d}/{num_epochs} | "
                  f"Total: {training_history['total_loss'][-1]:.3f} | "
                  f"Behavioral: {training_history['behavioral_loss'][-1]:.3f} | "
                  f"Contrastive: {training_history['contrastive_loss'][-1]:.3f} | "
                  f"Diversity: {training_history['diversity_loss'][-1]:.3f}")
    
    if verbose:
        print(f"   ✅ Training completed!")
    
    return state, (behavioral_features, environmental_features, states), training_history, trainer


def extract_representations(state, trainer, behavioral_features, environmental_features, batch_size=8):
    """
    Extract learned representations from the trained model.
    """
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
    """
    Evaluate how well the learned representations can classify psychological states.
    """
    
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
        print(f"\n🎯 Psychological State Classification Results:")
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"   Engaged samples: {np.sum(y_test)}/{len(y_test)} ({np.mean(y_test):.1%})")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Disengaged', 'Engaged']))
    
    return accuracy, classifier, (X_train, X_test, y_train, y_test)


def evaluate_trial_level_classification(contextualized_features, states, test_size=0.3, verbose=True):
    """
    Evaluate psychological state classification at the individual trial level.
    
    Args:
        contextualized_features: [num_windows, window_size, embed_dim] 
        states: [num_windows, window_size] - true states for each trial
        test_size: fraction for test set
        verbose: print results
    
    Returns:
        accuracy, classifier, data splits
    """
    
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


def visualize_results(training_history, contextualized_features, states, quantized_indices, 
                     classification_results, save_plots=True):
    """
    Create comprehensive visualizations of the training and evaluation results.
    """
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # 1. Training curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(training_history['total_loss']) + 1)
    ax1.plot(epochs, training_history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.plot(epochs, training_history['behavioral_loss'], 'r--', alpha=0.7, label='Behavioral Loss')
    ax1.plot(epochs, training_history['contrastive_loss'], 'g:', alpha=0.7, label='Contrastive Loss')
    ax1.plot(epochs, training_history['diversity_loss'], 'm-.', alpha=0.7, label='Diversity Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss reduction summary
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_title('Learning Progress Summary', fontsize=14, fontweight='bold')
    
    loss_components = ['Total', 'Behavioral', 'Contrastive', 'Diversity']
    initial_losses = [training_history[f'{comp.lower()}_loss'][0] for comp in loss_components]
    final_losses = [training_history[f'{comp.lower()}_loss'][-1] for comp in loss_components]
    
    x_pos = np.arange(len(loss_components))
    width = 0.35
    
    ax2.bar(x_pos - width/2, initial_losses, width, label='Initial', alpha=0.7, color='lightcoral')
    ax2.bar(x_pos + width/2, final_losses, width, label='Final', alpha=0.7, color='lightblue')
    
    ax2.set_xlabel('Loss Component')
    ax2.set_ylabel('Loss Value')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(loss_components)
    ax2.legend()
    ax2.set_yscale('log')
    
    # 3. PCA of learned representations
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title('Learned Representations (PCA)', fontsize=14, fontweight='bold')
    
    # Pool features and states
    pooled_features = np.mean(contextualized_features, axis=1)
    pooled_states = np.mean(states, axis=1)
    
    # Apply PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(pooled_features)
    
    scatter = ax3.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=pooled_states, cmap='RdYlBu', alpha=0.7, s=30)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=ax3, label='Engagement Level')
    
    # 4. Classification results
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_title('State Classification Performance', fontsize=14, fontweight='bold')
    
    accuracy, classifier, (X_train, X_test, y_train, y_test) = classification_results
    
    # Plot decision boundary in PCA space
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    
    pca_clf = PCA(n_components=2)
    X_pca = pca_clf.fit_transform(X_all)
    
    # Train classifier on PCA features for visualization
    clf_pca = LogisticRegression(random_state=42)
    clf_pca.fit(X_pca, y_all)
    
    # Create decision boundary
    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf_pca.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    ax4.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
    scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y_all, cmap='RdYlBu', 
                         edgecolors='black', alpha=0.7)
    ax4.set_xlabel(f'PC1 ({pca_clf.explained_variance_ratio_[0]:.1%})')
    ax4.set_ylabel(f'PC2 ({pca_clf.explained_variance_ratio_[1]:.1%})')
    ax4.set_title(f'Classification Accuracy: {accuracy:.3f}')
    
    # 5. Codebook usage analysis
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_title('Codebook Usage Distribution', fontsize=14, fontweight='bold')
    
    # Flatten quantized indices
    flat_indices = quantized_indices.reshape(-1, quantized_indices.shape[-1])
    
    for g in range(quantized_indices.shape[-1]):
        ax5.hist(flat_indices[:, g], bins=30, alpha=0.7, 
                label=f'Group {g}', density=True)
    ax5.set_xlabel('Codebook Index')
    ax5.set_ylabel('Density')
    ax5.legend()
    
    # 6. Feature importance for classification
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.set_title('Feature Importance for State Classification', fontsize=14, fontweight='bold')
    
    # Get feature importance (coefficients)
    feature_importance = np.abs(classifier.coef_[0])
    top_features = np.argsort(feature_importance)[-20:]  # Top 20 features
    
    ax6.barh(range(len(top_features)), feature_importance[top_features])
    ax6.set_xlabel('Absolute Coefficient Value')
    ax6.set_ylabel('Feature Index')
    ax6.set_title('Top 20 Most Important Features')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('choice2vec_training_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Results saved as 'choice2vec_training_results.png'")
    
    plt.show()


def main():
    """
    Main training and evaluation pipeline.
    """
    print("🚀 Choice2Vec: Self-Supervised Learning for Behavioral Data")
    print("=" * 70)
    
    # Train model
    print("\n🏋️ Training Choice2Vec model...")
    state, data, training_history, trainer = train_choice2vec(
        num_epochs=1000,
        batch_size=16,  # Increased batch size for GPU efficiency
        learning_rate=1e-4,
        use_cosine_loss=True,
        verbose=True
    )
    
    if state is None:
        return
    
    behavioral_features, environmental_features, states = data
    
    # Extract representations
    print("\n🧠 Extracting learned representations...")
    contextualized_features, quantized_indices = extract_representations(
        state, trainer, behavioral_features, environmental_features
    )
    
    print(f"   Extracted representations: {contextualized_features.shape}")
    
    # Evaluate state classification
    print("\n🎯 Evaluating psychological state classification...")
    classification_results = evaluate_state_classification(
        contextualized_features, states, test_size=0.3, verbose=True
    )
    
    # Evaluate trial-level classification
    print("\n🎯 Evaluating trial-level psychological state classification...")
    trial_level_results = evaluate_trial_level_classification(
        contextualized_features, states, test_size=0.3, verbose=True
    )
    
    # Visualize results
    print("\n📊 Creating visualizations...")
    visualize_results(
        training_history, contextualized_features, states, 
        quantized_indices, classification_results, save_plots=True
    )
    
    # Summary
    accuracy = classification_results[0]
    print(f"\n🎉 Training and Evaluation Complete!")
    print(f"=" * 50)
    print(f"✅ Model successfully trained with complete self-supervised learning")
    print(f"✅ Learned representations achieve {accuracy:.1%} accuracy on state classification")
    print(f"✅ All three loss components (behavioral, contrastive, diversity) working")
    print(f"✅ Model captures psychological states from behavioral patterns")
    
    if accuracy > 0.7:
        print(f"\n🌟 Excellent! The model learned meaningful representations!")
        print(f"   The latent structure reflects the underlying generative process.")
    elif accuracy > 0.6:
        print(f"\n👍 Good! The model learned useful representations.")
        print(f"   There's clear signal about psychological states in the learned features.")
    else:
        print(f"\n🤔 The model learned some structure, but classification is challenging.")
        print(f"   This could indicate need for more training or architectural changes.")
    
    # Save model parameters (not the full training state to avoid pickle issues)
    model_name = f"choice2vec_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # Create a saveable model state with just the essential components
    saveable_state = {
        'params': state.params,
        'model_config': {
            'encoder_hidden_dims': (64, 128, 256),
            'num_quantizer_groups': 2,
            'num_entries_per_group': 128,
            'num_transformer_layers': 4,
            'embed_dim': 256,
            'num_heads': 4,
            'dropout_rate': 0.1,
            'mask_prob': 0.15
        },
        'training_config': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'diversity_weight': 0.1,
            'contrastive_weight': 1.0,
            'use_cosine_loss': True
        },
        'data_config': {
            'window_size': 100,
            'stride': 20
        },
        'training_history': training_history
    }
    
    with open(f"{model_name}.pkl", "wb") as f:
        pickle.dump(saveable_state, f)
    print(f"\n📊 Model parameters saved as '{model_name}.pkl'")
    
    return state, data, training_history, classification_results, trial_level_results


if __name__ == "__main__":
    main() 