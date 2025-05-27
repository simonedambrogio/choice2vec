#!/usr/bin/env python3
"""
Training Script for Disentangled Choice2Vec

This script compares:
1. Standard Choice2Vec (baseline)
2. Disentangled Choice2Vec with multiple disentanglement techniques

It analyzes:
- PCA variance explained (should be more concentrated in fewer components)
- Factor independence (correlation between learned factors)
- Disentanglement quality metrics
- Downstream task performance
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from core.choice2vec_model import Choice2Vec, Choice2VecTrainer, prepare_behavioral_data
from disentangled_choice2vec import (
    DisentangledChoice2Vec, DisentangledChoice2VecTrainer, analyze_disentanglement
)

def extract_representations_disentangled(state, trainer, behavioral_features, environmental_features, batch_size=8):
    """Extract representations from disentangled model."""
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
    factor_projected_features = jnp.concatenate([out['factor_projected_features'] for out in all_outputs], axis=0)
    
    return np.array(contextualized_features), np.array(factor_projected_features)

def extract_factor_representations(factor_projected_features, factor_idx, embed_dim=256, num_factors=4):
    """Extract representations for a specific factor."""
    factor_dim = embed_dim // num_factors
    start_idx = factor_idx * factor_dim
    end_idx = (factor_idx + 1) * factor_dim
    
    return factor_projected_features[:, :, start_idx:end_idx]

def analyze_factor_specialization(factor_projected_features, df, window_size=100, stride=20):
    """
    Analyze what each factor captures by correlating with behavioral variables.
    
    Args:
        factor_projected_features: [num_windows, window_size, embed_dim] factor projections
        df: Original dataframe with behavioral variables
        window_size: Size of windows used in data preparation
        stride: Stride used in data preparation
    
    Returns:
        Dictionary with factor specialization analysis
    """
    print("\n🔍 Analyzing Factor Specialization...")
    print("=" * 50)
    
    # Prepare behavioral variables aligned with windows
    behavioral_variables = {}
    
    # Create windows for each behavioral variable to match the representation windows
    for var_name in ['psychological_state', 'trial_in_subtask', 'choice_correct', 'rt']:
        if var_name not in df.columns:
            continue
            
        var_windows = []
        for start_idx in range(0, len(df) - window_size + 1, stride):
            end_idx = start_idx + window_size
            if var_name == 'psychological_state':
                # Convert to binary and take mean engagement per window
                window_values = (df[var_name].iloc[start_idx:end_idx] == 'engaged').astype(float)
                var_windows.append(window_values.mean())
            else:
                # Take mean of continuous variables per window
                window_values = df[var_name].iloc[start_idx:end_idx]
                var_windows.append(window_values.mean())
        
        behavioral_variables[var_name] = np.array(var_windows)
    
    # Add learning progress if available
    if 'learning_progress' in df.columns:
        learning_windows = []
        for start_idx in range(0, len(df) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_values = df['learning_progress'].iloc[start_idx:end_idx]
            learning_windows.append(window_values.mean())
        behavioral_variables['learning_progress'] = np.array(learning_windows)
    
    print(f"   Analyzing {len(behavioral_variables)} behavioral variables across {len(factor_projected_features)} windows")
    
    # Analyze each factor
    factor_specialization = {}
    num_factors = 4
    
    for factor_idx in range(num_factors):
        print(f"\n📊 Factor {factor_idx + 1}:")
        
        # Extract factor representations and pool across sequence dimension
        factor_repr = extract_factor_representations(factor_projected_features, factor_idx)
        pooled_factor = np.mean(factor_repr, axis=1)  # [num_windows, factor_dim] -> [num_windows]
        
        # For analysis, take the mean across the factor dimension
        factor_signal = np.mean(pooled_factor, axis=1) if pooled_factor.ndim > 1 else pooled_factor
        
        correlations = {}
        for var_name, var_values in behavioral_variables.items():
            # Ensure same length
            min_len = min(len(factor_signal), len(var_values))
            factor_trimmed = factor_signal[:min_len]
            var_trimmed = var_values[:min_len]
            
            # Calculate correlation
            if len(np.unique(var_trimmed)) > 1 and np.std(factor_trimmed) > 1e-8:
                corr = np.corrcoef(factor_trimmed, var_trimmed)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            
            correlations[var_name] = abs(corr)  # Use absolute correlation
            print(f"   {var_name:20s}: r = {corr:+.3f} (|r| = {abs(corr):.3f})")
        
        # Find what this factor specializes in
        if correlations:
            specialized_variable = max(correlations.keys(), key=lambda k: correlations[k])
            max_correlation = correlations[specialized_variable]
            
            factor_specialization[f'Factor_{factor_idx + 1}'] = {
                'specializes_in': specialized_variable,
                'correlation': max_correlation,
                'all_correlations': correlations,
                'interpretation': get_factor_interpretation(specialized_variable, max_correlation)
            }
            
            print(f"   → Specializes in: {specialized_variable} (|r| = {max_correlation:.3f})")
            print(f"   → {get_factor_interpretation(specialized_variable, max_correlation)}")
        else:
            factor_specialization[f'Factor_{factor_idx + 1}'] = {
                'specializes_in': 'unknown',
                'correlation': 0.0,
                'all_correlations': {},
                'interpretation': 'No clear specialization found'
            }
    
    # Calculate factor independence (cross-correlations)
    print(f"\n🔗 Factor Independence Analysis:")
    factor_signals = []
    for factor_idx in range(num_factors):
        factor_repr = extract_factor_representations(factor_projected_features, factor_idx)
        pooled_factor = np.mean(factor_repr, axis=(1, 2))  # Pool across sequence and feature dims
        factor_signals.append(pooled_factor)
    
    # Compute cross-correlation matrix
    factor_correlations = np.corrcoef(factor_signals)
    
    # Report off-diagonal correlations (independence measure)
    off_diagonal_corrs = []
    for i in range(num_factors):
        for j in range(i + 1, num_factors):
            corr = factor_correlations[i, j]
            off_diagonal_corrs.append(abs(corr))
            print(f"   Factor {i+1} ↔ Factor {j+1}: r = {corr:+.3f}")
    
    mean_cross_correlation = np.mean(off_diagonal_corrs)
    independence_score = 1.0 - mean_cross_correlation
    
    print(f"\n📈 Independence Summary:")
    print(f"   Mean cross-correlation: {mean_cross_correlation:.3f}")
    print(f"   Independence score: {independence_score:.3f}")
    
    if independence_score > 0.7:
        print(f"   ✅ Excellent factor independence!")
    elif independence_score > 0.5:
        print(f"   👍 Good factor independence")
    else:
        print(f"   ⚠️ Factors may be entangled")
    
    # Add summary to results
    factor_specialization['summary'] = {
        'mean_cross_correlation': mean_cross_correlation,
        'independence_score': independence_score,
        'factor_correlations_matrix': factor_correlations.tolist()
    }
    
    return factor_specialization

def get_factor_interpretation(variable_name, correlation):
    """Get human-readable interpretation of what a factor captures."""
    interpretations = {
        'psychological_state': f"Psychological engagement states (engaged vs disengaged)",
        'trial_in_subtask': f"Task structure and trial position within subtasks",
        'choice_correct': f"Performance patterns and accuracy dynamics", 
        'rt': f"Response time patterns and decision speed",
        'learning_progress': f"Learning dynamics and skill acquisition"
    }
    
    base_interpretation = interpretations.get(variable_name, f"Patterns in {variable_name}")
    
    if correlation > 0.7:
        strength = "Strong"
    elif correlation > 0.5:
        strength = "Moderate"
    elif correlation > 0.3:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    return f"{strength} specialization in {base_interpretation.lower()}"

def analyze_factor_specific_clustering(factor_projected_features, states, factor_specialization):
    """
    Perform clustering on individual factors, especially the one that specializes in psychological states.
    """
    print("\n🎯 Factor-Specific Clustering Analysis")
    print("=" * 50)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, accuracy_score
    
    # Find the factor that best captures psychological states
    psych_factor_idx = None
    best_psych_correlation = 0
    
    for factor_name, factor_info in factor_specialization.items():
        if factor_name != 'summary' and factor_info['specializes_in'] == 'psychological_state':
            correlation = factor_info['correlation']
            if correlation > best_psych_correlation:
                best_psych_correlation = correlation
                psych_factor_idx = int(factor_name.split('_')[1]) - 1  # Convert to 0-indexed
    
    if psych_factor_idx is not None:
        print(f"   Found psychology-specialized factor: Factor {psych_factor_idx + 1} (|r| = {best_psych_correlation:.3f})")
        
        # Extract and cluster the psychology-specialized factor
        psych_factor_repr = extract_factor_representations(factor_projected_features, psych_factor_idx)
        pooled_psych_factor = np.mean(psych_factor_repr, axis=(1, 2))  # Pool to [num_windows]
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pooled_psych_factor.reshape(-1, 1))
        
        # Convert true states to binary
        pooled_states = np.mean(states, axis=1)
        true_binary = (pooled_states > 0.5).astype(int)
        
        # Calculate accuracy (try both label assignments)
        acc1 = accuracy_score(true_binary, cluster_labels)
        acc2 = accuracy_score(true_binary, 1 - cluster_labels)
        factor_accuracy = max(acc1, acc2)
        
        # Calculate ARI
        ari = adjusted_rand_score(true_binary, cluster_labels)
        
        print(f"   Psychology Factor Clustering:")
        print(f"     • Accuracy: {factor_accuracy:.3f}")
        print(f"     • ARI: {ari:.3f}")
        
        return {
            'psychology_factor_idx': psych_factor_idx,
            'psychology_factor_accuracy': factor_accuracy,
            'psychology_factor_ari': ari,
            'psychology_factor_correlation': best_psych_correlation
        }
    else:
        print("   ⚠️ No factor strongly specializes in psychological states")
        print("   This suggests the disentanglement may not have worked as expected")
        
        # Still try clustering on all factors to see which works best
        best_factor_accuracy = 0
        best_factor_idx = 0
        
        for factor_idx in range(4):
            factor_repr = extract_factor_representations(factor_projected_features, factor_idx)
            pooled_factor = np.mean(factor_repr, axis=(1, 2))
            
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pooled_factor.reshape(-1, 1))
            
            pooled_states = np.mean(states, axis=1)
            true_binary = (pooled_states > 0.5).astype(int)
            
            acc1 = accuracy_score(true_binary, cluster_labels)
            acc2 = accuracy_score(true_binary, 1 - cluster_labels)
            factor_accuracy = max(acc1, acc2)
            
            print(f"     Factor {factor_idx + 1} clustering accuracy: {factor_accuracy:.3f}")
            
            if factor_accuracy > best_factor_accuracy:
                best_factor_accuracy = factor_accuracy
                best_factor_idx = factor_idx
        
        print(f"   Best factor for clustering: Factor {best_factor_idx + 1} (acc = {best_factor_accuracy:.3f})")
        
        return {
            'psychology_factor_idx': None,
            'best_clustering_factor_idx': best_factor_idx,
            'best_clustering_accuracy': best_factor_accuracy,
            'psychology_factor_correlation': 0.0
        }

def train_standard_choice2vec(behavioral_features, environmental_features, states, num_epochs=1000):
    """Train standard Choice2Vec model."""
    print("🔄 Training Standard Choice2Vec...")
    
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
        use_cosine_loss=True,
        temperature=0.1
    )
    
    # Initialize training state
    rng = jax.random.PRNGKey(42)
    behavioral_shape = (1, behavioral_features.shape[1], behavioral_features.shape[2])
    environmental_shape = (1, environmental_features.shape[1], environmental_features.shape[2])
    
    state = trainer.create_train_state(rng, behavioral_shape, environmental_shape)
    
    # Training loop
    behavioral_jax = jnp.array(behavioral_features)
    environmental_jax = jnp.array(environmental_features)
    
    training_history = {
        'total_loss': [],
        'behavioral_loss': [],
        'contrastive_loss': [],
        'diversity_loss': []
    }
    
    batch_size = 16
    
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
        
        if epoch % 200 == 0 or epoch == num_epochs - 1:
            print(f"   Epoch {epoch + 1:4d}/{num_epochs} | "
                  f"Total: {training_history['total_loss'][-1]:.3f} | "
                  f"Behavioral: {training_history['behavioral_loss'][-1]:.3f} | "
                  f"Contrastive: {training_history['contrastive_loss'][-1]:.3f}")
    
    return state, trainer, training_history

def train_disentangled_choice2vec(behavioral_features, environmental_features, states, num_epochs=1000):
    """Train disentangled Choice2Vec model."""
    print("🧠 Training Disentangled Choice2Vec...")
    
    model = DisentangledChoice2Vec(
        encoder_hidden_dims=(64, 128, 256),
        num_quantizer_groups=4,  # More groups for better factorization
        num_entries_per_group=64,  # Smaller codebooks per group
        num_transformer_layers=4,
        embed_dim=256,
        num_heads=4,
        dropout_rate=0.1,
        mask_prob=0.15,
        beta=4.0  # β-VAE strength
    )
    
    trainer = DisentangledChoice2VecTrainer(
        model=model,
        learning_rate=1e-4,
        weight_decay=0.01,
        diversity_weight=0.1,
        contrastive_weight=1.0,
        factor_contrastive_weight=0.5,  # Factor-wise contrastive learning
        mi_weight=0.1,  # Mutual information minimization
        orthogonality_weight=0.05,  # Orthogonality constraint
        commitment_weight=1.0,  # β-VAE commitment loss
        temperature=0.1
    )
    
    # Initialize training state
    rng = jax.random.PRNGKey(42)
    behavioral_shape = (1, behavioral_features.shape[1], behavioral_features.shape[2])
    environmental_shape = (1, environmental_features.shape[1], environmental_features.shape[2])
    
    state = trainer.create_train_state(rng, behavioral_shape, environmental_shape)
    
    # Training loop
    behavioral_jax = jnp.array(behavioral_features)
    environmental_jax = jnp.array(environmental_features)
    
    training_history = {
        'total_loss': [],
        'behavioral_loss': [],
        'global_contrastive_loss': [],
        'factor_contrastive_loss': [],
        'mi_loss': [],
        'orthogonality_loss': [],
        'commitment_loss': [],
        'diversity_loss': []
    }
    
    batch_size = 16
    
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
        
        if epoch % 200 == 0 or epoch == num_epochs - 1:
            print(f"   Epoch {epoch + 1:4d}/{num_epochs} | "
                  f"Total: {training_history['total_loss'][-1]:.3f} | "
                  f"Behavioral: {training_history['behavioral_loss'][-1]:.3f} | "
                  f"MI: {training_history['mi_loss'][-1]:.3f} | "
                  f"Ortho: {training_history['orthogonality_loss'][-1]:.3f}")
    
    return state, trainer, training_history

def analyze_pca_variance(representations, model_name):
    """Analyze PCA variance explained by different numbers of components."""
    print(f"\n📊 PCA Analysis for {model_name}:")
    
    # Pool representations across sequence dimension
    pooled_repr = np.mean(representations, axis=1)  # [num_windows, embed_dim]
    
    # Fit PCA
    pca = PCA()
    pca.fit(pooled_repr)
    
    # Calculate cumulative variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for different variance thresholds
    n_90 = np.argmax(cumvar >= 0.90) + 1
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_99 = np.argmax(cumvar >= 0.99) + 1
    
    print(f"   Components for 90% variance: {n_90}")
    print(f"   Components for 95% variance: {n_95}")
    print(f"   Components for 99% variance: {n_99}")
    print(f"   First PC explains: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"   First 3 PCs explain: {cumvar[2]:.3f}")
    
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumvar,
        'n_components_90': n_90,
        'n_components_95': n_95,
        'n_components_99': n_99,
        'first_pc_variance': pca.explained_variance_ratio_[0],
        'first_3_pcs_variance': cumvar[2]
    }

def evaluate_downstream_task(representations, states, model_name):
    """Evaluate psychological state classification performance."""
    print(f"\n🎯 Downstream Task Evaluation for {model_name}:")
    
    # Pool representations
    pooled_repr = np.mean(representations, axis=1)
    pooled_states = np.mean(states, axis=1)
    binary_states = (pooled_states > 0.5).astype(int)
    
    # Split data
    n_samples = len(pooled_repr)
    n_train = int(n_samples * 0.7)
    
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_test = pooled_repr[train_idx], pooled_repr[test_idx]
    y_train, y_test = binary_states[train_idx], binary_states[test_idx]
    
    # Train classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   Test Accuracy: {accuracy:.3f}")
    print(f"   Train/Test split: {len(X_train)}/{len(X_test)}")
    
    return accuracy

def create_comparison_visualization(standard_results, disentangled_results, save_path=None):
    """Create comprehensive comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PCA Variance Comparison
    ax = axes[0, 0]
    n_components = min(50, len(standard_results['pca']['explained_variance_ratio']))
    
    ax.plot(range(1, n_components + 1), 
            standard_results['pca']['cumulative_variance'][:n_components], 
            'b-', label='Standard Choice2Vec', linewidth=2)
    ax.plot(range(1, n_components + 1), 
            disentangled_results['pca']['cumulative_variance'][:n_components], 
            'r-', label='Disentangled Choice2Vec', linewidth=2)
    
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95% Variance')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('PCA Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. First Few PC Variance
    ax = axes[0, 1]
    n_show = 10
    
    x_pos = np.arange(n_show)
    width = 0.35
    
    ax.bar(x_pos - width/2, standard_results['pca']['explained_variance_ratio'][:n_show], 
           width, label='Standard', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, disentangled_results['pca']['explained_variance_ratio'][:n_show], 
           width, label='Disentangled', alpha=0.7, color='red')
    
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('Individual PC Variance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_show)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Disentanglement Metrics
    ax = axes[0, 2]
    
    if 'disentanglement' in disentangled_results:
        metrics = disentangled_results['disentanglement']
        
        metric_names = ['Independence\n(1-correlation)', 'Best Factor\nAccuracy', 'Disentanglement\nScore']
        metric_values = [
            1.0 - metrics['mean_factor_correlation'],
            metrics['best_factor_accuracy'],
            metrics['disentanglement_score']
        ]
        
        bars = ax.bar(metric_names, metric_values, alpha=0.7, color=['green', 'orange', 'purple'])
        ax.set_ylabel('Score')
        ax.set_title('Disentanglement Quality')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Training Loss Comparison (Standard)
    ax = axes[1, 0]
    
    epochs = range(1, len(standard_results['training_history']['total_loss']) + 1)
    ax.plot(epochs, standard_results['training_history']['total_loss'], 'b-', label='Total', linewidth=2)
    ax.plot(epochs, standard_results['training_history']['behavioral_loss'], 'r--', label='Behavioral')
    ax.plot(epochs, standard_results['training_history']['contrastive_loss'], 'g:', label='Contrastive')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Standard Choice2Vec Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Training Loss Comparison (Disentangled)
    ax = axes[1, 1]
    
    epochs = range(1, len(disentangled_results['training_history']['total_loss']) + 1)
    ax.plot(epochs, disentangled_results['training_history']['total_loss'], 'b-', label='Total', linewidth=2)
    ax.plot(epochs, disentangled_results['training_history']['behavioral_loss'], 'r--', label='Behavioral')
    ax.plot(epochs, disentangled_results['training_history']['mi_loss'], 'g:', label='MI Loss')
    ax.plot(epochs, disentangled_results['training_history']['orthogonality_loss'], 'm-.', label='Orthogonality')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Disentangled Choice2Vec Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Performance Summary
    ax = axes[1, 2]
    
    models = ['Standard', 'Disentangled']
    accuracies = [standard_results['accuracy'], disentangled_results['accuracy']]
    n_95_components = [standard_results['pca']['n_components_95'], 
                      disentangled_results['pca']['n_components_95']]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.7, color='skyblue')
    bars2 = ax2.bar(x_pos + width/2, n_95_components, width, label='PCs for 95%', alpha=0.7, color='lightcoral')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Classification Accuracy', color='blue')
    ax2.set_ylabel('Components for 95% Variance', color='red')
    ax.set_title('Performance Summary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    
    # Add value labels
    for bar, value in zip(bars1, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', color='blue')
    
    for bar, value in zip(bars2, n_95_components):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}', ha='center', va='bottom', color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison visualization saved to {save_path}")
    
    plt.show()

def main():
    """Main comparison pipeline."""
    print("🔬 Disentangled Choice2Vec Comparison")
    print("=" * 70)
    
    # Check if data exists - prefer psychological data format
    if os.path.exists('results/psychological_behavioral_data.csv'):
        print("📊 Loading psychological behavioral data (choice_correct format)...")
        df = pd.read_csv('results/psychological_behavioral_data.csv')
    elif os.path.exists('results/behavioral_data.csv'):
        print("📊 Loading standard behavioral data (choice format)...")
        df = pd.read_csv('results/behavioral_data.csv')
    else:
        print("❌ Error: No behavioral data found!")
        print("   Please run 'python data_generation/generate_psychological_data.py' first.")
        return
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=100, stride=20
    )
    
    print(f"   Dataset: {len(df)} trials, {len(behavioral_features)} windows")
    
    # Train both models
    num_epochs = 1000
    
    # 1. Train standard Choice2Vec
    standard_state, standard_trainer, standard_history = train_standard_choice2vec(
        behavioral_features, environmental_features, states, num_epochs
    )
    
    # 2. Train disentangled Choice2Vec
    disentangled_state, disentangled_trainer, disentangled_history = train_disentangled_choice2vec(
        behavioral_features, environmental_features, states, num_epochs
    )
    
    # Extract representations
    print("\n🧠 Extracting representations...")
    
    # Standard model (use existing function)
    from standard_choice2vec.train_choice2vec import extract_representations
    standard_repr, _ = extract_representations(
        standard_state, standard_trainer, behavioral_features, environmental_features
    )
    
    # Disentangled model
    disentangled_repr, factor_projected_features = extract_representations_disentangled(
        disentangled_state, disentangled_trainer, behavioral_features, environmental_features
    )
    
    print(f"   Standard representations: {standard_repr.shape}")
    print(f"   Disentangled representations: {disentangled_repr.shape}")
    print(f"   Factor projected features: {factor_projected_features.shape}")
    
    # Analyze representations
    print("\n📈 Analyzing representation quality...")
    
    # PCA analysis
    standard_pca = analyze_pca_variance(standard_repr, "Standard Choice2Vec")
    disentangled_pca = analyze_pca_variance(disentangled_repr, "Disentangled Choice2Vec")
    
    # Downstream task evaluation
    standard_accuracy = evaluate_downstream_task(standard_repr, states, "Standard Choice2Vec")
    disentangled_accuracy = evaluate_downstream_task(disentangled_repr, states, "Disentangled Choice2Vec")
    
    # Factor specialization analysis (new!)
    factor_specialization = analyze_factor_specialization(
        factor_projected_features, df, window_size=100, stride=20
    )
    
    # Factor-specific clustering analysis
    factor_clustering_results = analyze_factor_specific_clustering(
        factor_projected_features, states, factor_specialization
    )
    
    # Disentanglement analysis (only for disentangled model)
    print("\n🔍 Analyzing disentanglement quality...")
    disentanglement_metrics = analyze_disentanglement(
        jnp.array(disentangled_repr), 
        jnp.array(states), 
        num_groups=4
    )
    
    print("   Disentanglement Metrics:")
    for key, value in disentanglement_metrics.items():
        print(f"     {key}: {value:.3f}")
    
    # Compile results
    standard_results = {
        'pca': standard_pca,
        'accuracy': standard_accuracy,
        'training_history': standard_history
    }
    
    disentangled_results = {
        'pca': disentangled_pca,
        'accuracy': disentangled_accuracy,
        'training_history': disentangled_history,
        'disentanglement': disentanglement_metrics
    }
    
    # Create visualization
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f'results/disentanglement_comparison_{timestamp}.png'
    
    create_comparison_visualization(standard_results, disentangled_results, save_path)
    
    # Summary
    print(f"\n🎉 COMPARISON SUMMARY")
    print(f"=" * 50)
    print(f"Standard Choice2Vec:")
    print(f"   • Classification Accuracy: {standard_accuracy:.3f}")
    print(f"   • Components for 95% variance: {standard_pca['n_components_95']}")
    print(f"   • First PC explains: {standard_pca['first_pc_variance']:.3f}")
    
    print(f"\nDisentangled Choice2Vec:")
    print(f"   • Classification Accuracy: {disentangled_accuracy:.3f}")
    print(f"   • Components for 95% variance: {disentangled_pca['n_components_95']}")
    print(f"   • First PC explains: {disentangled_pca['first_pc_variance']:.3f}")
    print(f"   • Factor Independence: {1.0 - disentanglement_metrics['mean_factor_correlation']:.3f}")
    print(f"   • Disentanglement Score: {disentanglement_metrics['disentanglement_score']:.3f}")
    
    print(f"\n🧠 Factor Specialization Summary:")
    for factor_name, factor_info in factor_specialization.items():
        if factor_name != 'summary':
            specialization = factor_info['specializes_in']
            correlation = factor_info['correlation']
            print(f"   • {factor_name}: {specialization} (|r| = {correlation:.3f})")
    
    independence_score = factor_specialization['summary']['independence_score']
    print(f"   • Overall Independence Score: {independence_score:.3f}")
    
    print(f"\n🎯 Factor-Specific Clustering Results:")
    if factor_clustering_results['psychology_factor_idx'] is not None:
        psych_factor_idx = factor_clustering_results['psychology_factor_idx']
        psych_accuracy = factor_clustering_results['psychology_factor_accuracy']
        print(f"   • Psychology Factor {psych_factor_idx + 1} Clustering: {psych_accuracy:.3f} accuracy")
        print(f"   • This factor specifically captures psychological states!")
    else:
        best_idx = factor_clustering_results.get('best_clustering_factor_idx', 0)
        best_acc = factor_clustering_results.get('best_clustering_accuracy', 0)
        print(f"   • Best clustering factor: Factor {best_idx + 1} ({best_acc:.3f} accuracy)")
        print(f"   • No factor strongly specializes in psychological states")
    
    # Determine winner
    improvement_pca = standard_pca['n_components_95'] - disentangled_pca['n_components_95']
    improvement_acc = disentangled_accuracy - standard_accuracy
    
    print(f"\n🏆 RESULTS:")
    if improvement_pca > 0:
        print(f"   ✅ Disentanglement IMPROVED: {improvement_pca} fewer PCs needed for 95% variance")
    else:
        print(f"   ❌ Disentanglement did not improve PCA efficiency")
    
    if improvement_acc > 0.01:
        print(f"   ✅ Classification IMPROVED: +{improvement_acc:.3f} accuracy")
    elif improvement_acc > -0.01:
        print(f"   ➖ Classification MAINTAINED: {improvement_acc:+.3f} accuracy")
    else:
        print(f"   ❌ Classification DEGRADED: {improvement_acc:.3f} accuracy")
    
    if disentanglement_metrics['disentanglement_score'] > 0.5:
        print(f"   ✅ Good disentanglement quality achieved!")
    else:
        print(f"   ⚠️ Disentanglement quality could be improved")
    
    # Factor specialization insights
    print(f"\n🧠 FACTOR SPECIALIZATION INSIGHTS:")
    psych_factor_found = factor_clustering_results['psychology_factor_idx'] is not None
    if psych_factor_found:
        psych_factor_idx = factor_clustering_results['psychology_factor_idx']
        psych_accuracy = factor_clustering_results['psychology_factor_accuracy']
        psych_correlation = factor_clustering_results['psychology_factor_correlation']
        print(f"   ✅ Factor {psych_factor_idx + 1} successfully specializes in psychological states!")
        print(f"   ✅ Psychology-specific clustering: {psych_accuracy:.1%} accuracy")
        print(f"   ✅ Strong correlation with engagement: |r| = {psych_correlation:.3f}")
        
        if psych_accuracy > disentangled_accuracy:
            improvement = psych_accuracy - disentangled_accuracy
            print(f"   🎯 Factor-specific clustering IMPROVED by {improvement:.1%} over full representation!")
        
    else:
        print(f"   ❌ No factor strongly specializes in psychological states")
        print(f"   ❌ Disentanglement may not have separated psychological from other factors")
        print(f"   💡 Consider: stronger regularization, more training, or different architecture")
    
    if independence_score > 0.7:
        print(f"   ✅ Factors are well-separated (independence = {independence_score:.3f})")
    else:
        print(f"   ⚠️ Factors may be entangled (independence = {independence_score:.3f})")

if __name__ == "__main__":
    main() 