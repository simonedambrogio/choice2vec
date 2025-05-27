#!/usr/bin/env python3
"""
Unsupervised Psychological State Discovery with Choice2Vec
Using Psychological Data (Correct/Incorrect Choices)

This script trains Choice2Vec with InfoNCE loss on the new psychological data format,
then performs unsupervised clustering to discover psychological states without using
any labels during clustering.

Key improvements:
1. Uses correct/incorrect choices (not left/right)
2. No value_difference available (prevents trivial solutions)
3. Model must learn psychological patterns from choice accuracy and response times
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from choice2vec_model import prepare_behavioral_data

def train_extended_infonce_psychological():
    """
    Train Choice2Vec with InfoNCE loss for 150,000 epochs on psychological data.
    Save checkpoints every 10,000 epochs with weights, representations, and clustering results.
    """
    print("🧠 Training Choice2Vec with InfoNCE on Psychological Data")
    print("Focus: Correct/Incorrect choices (no value_difference)")
    print("Training for 150,000 epochs with checkpoints every 10,000 epochs")
    print("=" * 80)
    
    # Check if psychological data exists
    if not os.path.exists('psychological_behavioral_data.csv'):
        print("❌ Error: 'psychological_behavioral_data.csv' not found!")
        print("   Please run 'python generate_psychological_data.py' first.")
        return None, None, None, None
    
    # Create main output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    main_output_dir = f'training_checkpoints_{timestamp}'
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"📁 Created output directory: {main_output_dir}")
    
    # Load psychological data
    df = pd.read_csv('psychological_behavioral_data.csv')
    print(f"📊 Loaded psychological dataset:")
    print(f"   {len(df)} trials across {df['subtask'].nunique()} subtasks")
    print(f"   Psychological states: {df['psychological_state'].value_counts().to_dict()}")
    print(f"   Overall accuracy: {df['choice_correct'].mean():.3f}")
    print(f"   Engaged accuracy: {df[df['psychological_state'] == 'engaged']['choice_correct'].mean():.3f}")
    print(f"   Disengaged accuracy: {df[df['psychological_state'] == 'disengaged']['choice_correct'].mean():.3f}")
    
    # Import training components directly
    import jax
    import jax.numpy as jnp
    import pickle
    from choice2vec_model import Choice2Vec, Choice2VecTrainer, prepare_behavioral_data
    
    print("📊 Loading and preparing data...")
    
    # Prepare data directly from the psychological dataframe
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=100, stride=20  # Larger windows, more data per sequence
    )
    
    print(f"   Loaded {len(df)} trials")
    print(f"   Created {len(behavioral_features)} windows")
    print(f"   Behavioral features: {behavioral_features.shape} [choice_correct, rt]")
    print(f"   Environmental features: {environmental_features.shape} [trial_in_subtask, subtask]")
    
    # Initialize model
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
        learning_rate=1e-4,
        weight_decay=0.01,
        diversity_weight=0.1,
        contrastive_weight=1.0,
        use_cosine_loss=False,
        use_wav2vec2_loss=True,
        temperature=0.1,
        num_negatives=100
    )
    
    # Initialize training state
    rng = jax.random.PRNGKey(42)
    behavioral_shape = (1, behavioral_features.shape[1], behavioral_features.shape[2])
    environmental_shape = (1, environmental_features.shape[1], environmental_features.shape[2])
    
    state = trainer.create_train_state(rng, behavioral_shape, environmental_shape)
    
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
    
    num_epochs = 150000
    batch_size = 16
    checkpoint_interval = 10000
    
    print(f"\n🏋️ Training for {num_epochs} epochs...")
    print(f"   Checkpoints every {checkpoint_interval} epochs")
    
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
        
        # Print progress
        if epoch % 1000 == 0 or epoch == num_epochs - 1:
            print(f"   Epoch {epoch + 1:6d}/{num_epochs} | "
                  f"Total: {training_history['total_loss'][-1]:.3f} | "
                  f"Behavioral: {training_history['behavioral_loss'][-1]:.3f} | "
                  f"Contrastive: {training_history['contrastive_loss'][-1]:.3f} | "
                  f"Diversity: {training_history['diversity_loss'][-1]:.3f}")
        
        # Save checkpoint every 10,000 epochs
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            checkpoint_epoch = epoch + 1
            print(f"\n💾 Saving checkpoint at epoch {checkpoint_epoch}...")
            
            # Create checkpoint subdirectory
            checkpoint_dir = os.path.join(main_output_dir, f'checkpoint_epoch_{checkpoint_epoch:06d}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 1. Save model weights and states
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
                    'use_cosine_loss': False,
                    'use_wav2vec2_loss': True,
                    'temperature': 0.1,
                    'num_negatives': 100
                },
                'data_config': {
                    'window_size': 100,
                    'stride': 20
                },
                'training_history': training_history,
                'epoch': checkpoint_epoch
            }
            
            weights_path = os.path.join(checkpoint_dir, 'model_weights.pkl')
            with open(weights_path, 'wb') as f:
                pickle.dump(saveable_state, f)
            print(f"   ✅ Saved model weights to {weights_path}")
            
            # 2. Extract trial-level representations and perform clustering
            print(f"   🧠 Extracting representations and performing clustering...")
            trial_representations, true_states = extract_trial_representations_psychological(state, trainer, df)
            
            # Perform clustering
            clustering_results, X_scaled, true_labels = perform_unsupervised_clustering_psychological(
                trial_representations, true_states, df[['trial_in_subtask', 'subtask', 'choice_correct', 'rt']]
            )
            
            # 3. Save comprehensive results object
            results_object = {
                'epoch': checkpoint_epoch,
                'true_states': true_states,  # Original string labels ('engaged'/'disengaged')
                'true_labels_binary': true_labels,  # Binary labels (0/1)
                'representations_256d': trial_representations,  # 256D representations from model
                'clustering_results': clustering_results,  # All clustering algorithm results
                'trial_info': df[['trial_in_subtask', 'subtask', 'choice_correct', 'rt']].to_dict('records'),
                'training_metrics': {
                    'total_loss': training_history['total_loss'][-1],
                    'behavioral_loss': training_history['behavioral_loss'][-1],
                    'contrastive_loss': training_history['contrastive_loss'][-1],
                    'diversity_loss': training_history['diversity_loss'][-1]
                }
            }
            
            results_path = os.path.join(checkpoint_dir, 'clustering_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results_object, f)
            print(f"   ✅ Saved clustering results to {results_path}")
            
            # 4. Create and save visualization
            print(f"   📊 Creating visualization...")
            X_pca, X_tsne = create_checkpoint_visualization(
                trial_representations, clustering_results, true_labels, 
                df[['trial_in_subtask', 'subtask', 'choice_correct', 'rt']], 
                training_history, checkpoint_epoch, checkpoint_dir
            )
            
            print(f"   ✅ Checkpoint {checkpoint_epoch} completed in {checkpoint_dir}")
    
    print(f"\n✅ Training completed!")
    print(f"   Final contrastive loss: {training_history['contrastive_loss'][-1]:.3f}")
    print(f"   Final behavioral loss: {training_history['behavioral_loss'][-1]:.3f}")
    print(f"   Final total loss: {training_history['total_loss'][-1]:.3f}")
    print(f"   All checkpoints saved in: {main_output_dir}")
    
    return state, (behavioral_features, environmental_features, states), training_history, trainer

def extract_trial_representations_psychological(state, trainer, df):
    """
    Extract representations for individual trials from psychological data.
    """
    print("\n🧠 Extracting trial-level representations...")
    
    # Prepare data with single-trial windows (window_size=1, stride=1)
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=1, stride=1
    )
    
    print(f"   Created {len(behavioral_features)} single-trial windows")
    print(f"   Behavioral features: {behavioral_features.shape} [choice_correct, rt]")
    print(f"   Environmental features: {environmental_features.shape} [trial_in_subtask, subtask]")
    
    # Extract representations
    contextualized_features, quantized_indices = extract_representations(
        state, trainer, behavioral_features, environmental_features
    )
    
    # Since each window has only 1 trial, we can directly use the representations
    # Shape: (n_trials, embed_dim)
    trial_representations = contextualized_features.squeeze(axis=1)  # Remove sequence dimension
    
    print(f"   Trial representations shape: {trial_representations.shape}")
    
    # Get the true psychological states from the original dataframe
    # Since we used window_size=1 and stride=1, we should have one representation per trial
    true_psychological_states = df['psychological_state'].values
    
    print(f"   True states shape: {true_psychological_states.shape}")
    print(f"   Engaged trials: {np.sum(true_psychological_states == 'engaged')} ({np.mean(true_psychological_states == 'engaged'):.1%})")
    print(f"   Disengaged trials: {np.sum(true_psychological_states == 'disengaged')} ({np.mean(true_psychological_states == 'disengaged'):.1%})")
    
    return trial_representations, true_psychological_states

def perform_unsupervised_clustering_psychological(representations, true_states, trial_info):
    """
    Perform multiple unsupervised clustering algorithms and evaluate against ground truth.
    """
    print("\n🎯 Performing Unsupervised Clustering on Psychological Representations")
    print("=" * 70)
    
    # Diagnostic analysis of representations
    print("🔍 Representation Quality Analysis:")
    true_binary = (true_states == 'engaged').astype(int)
    engaged_mask = true_binary == 1
    disengaged_mask = true_binary == 0
    
    engaged_reps = representations[engaged_mask]
    disengaged_reps = representations[disengaged_mask]
    
    print(f"   Engaged trials: {np.sum(engaged_mask)} ({np.mean(engaged_mask):.1%})")
    print(f"   Disengaged trials: {np.sum(disengaged_mask)} ({np.mean(disengaged_mask):.1%})")
    
    # Calculate mean representations for each state
    engaged_mean = np.mean(engaged_reps, axis=0)
    disengaged_mean = np.mean(disengaged_reps, axis=0)
    
    # Calculate distances
    euclidean_dist = np.linalg.norm(engaged_mean - disengaged_mean)
    cosine_sim = np.dot(engaged_mean, disengaged_mean) / (np.linalg.norm(engaged_mean) * np.linalg.norm(disengaged_mean))
    
    print(f"   Mean representation distance: {euclidean_dist:.3f}")
    print(f"   Mean representation cosine similarity: {cosine_sim:.3f}")
    
    # Calculate within-group vs between-group distances
    engaged_distances = []
    disengaged_distances = []
    between_distances = []
    
    # Sample for efficiency
    n_sample = min(500, len(engaged_reps), len(disengaged_reps))
    engaged_sample = engaged_reps[:n_sample]
    disengaged_sample = disengaged_reps[:n_sample]
    
    for i in range(min(100, n_sample)):
        for j in range(i+1, min(100, n_sample)):
            engaged_distances.append(np.linalg.norm(engaged_sample[i] - engaged_sample[j]))
            disengaged_distances.append(np.linalg.norm(disengaged_sample[i] - disengaged_sample[j]))
        
        for j in range(min(100, n_sample)):
            between_distances.append(np.linalg.norm(engaged_sample[i] - disengaged_sample[j]))
    
    print(f"   Within-engaged distance: {np.mean(engaged_distances):.3f} ± {np.std(engaged_distances):.3f}")
    print(f"   Within-disengaged distance: {np.mean(disengaged_distances):.3f} ± {np.std(disengaged_distances):.3f}")
    print(f"   Between-group distance: {np.mean(between_distances):.3f} ± {np.std(between_distances):.3f}")
    
    # Separability ratio (higher is better for clustering)
    separability = np.mean(between_distances) / np.mean(engaged_distances + disengaged_distances)
    print(f"   Separability ratio: {separability:.3f} (>1.0 indicates good separation)")
    
    # Standardize features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(representations)
    
    # Define clustering algorithms
    clustering_algorithms = {
        'K-Means (k=2)': KMeans(n_clusters=2, random_state=42, n_init=10),
        'Gaussian Mixture (k=2)': GaussianMixture(n_components=2, random_state=42),
        'Agglomerative (k=2)': AgglomerativeClustering(n_clusters=2, linkage='ward'),
        'Agglomerative Average (k=2)': AgglomerativeClustering(n_clusters=2, linkage='average'),
        'DBSCAN (eps=0.1)': DBSCAN(eps=0.1, min_samples=10),
        'DBSCAN (eps=0.2)': DBSCAN(eps=0.2, min_samples=10),
        'DBSCAN (eps=0.5)': DBSCAN(eps=0.5, min_samples=20),
        'DBSCAN (eps=1.0)': DBSCAN(eps=1.0, min_samples=50),
    }
    
    results = {}
    
    for name, algorithm in clustering_algorithms.items():
        print(f"\n📊 Running {name}...")
        
        try:
            # Fit clustering algorithm
            if hasattr(algorithm, 'predict'):
                cluster_labels = algorithm.fit_predict(X_scaled)
            else:
                cluster_labels = algorithm.fit(X_scaled).labels_
            
            # Handle DBSCAN noise points (-1 labels)
            if name.startswith('DBSCAN'):
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = np.sum(cluster_labels == -1)
                print(f"   Found {n_clusters} clusters ({n_noise} noise points)")
                
                if n_clusters < 2:
                    print("   ⚠️ Too few clusters found, skipping...")
                    continue
            
            # Convert true states to binary labels
            true_labels = (true_states == 'engaged').astype(int)
            
            # Calculate clustering metrics
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            
            # Calculate silhouette score (measure of cluster quality)
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(X_scaled, cluster_labels)
            else:
                silhouette = -1
            
            # Calculate accuracy by finding best label assignment
            accuracy = calculate_clustering_accuracy(true_labels, cluster_labels)
            
            results[name] = {
                'cluster_labels': cluster_labels,
                'ari': ari,
                'nmi': nmi,
                'silhouette': silhouette,
                'accuracy': accuracy,
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            }
            
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   ARI: {ari:.3f}")
            print(f"   NMI: {nmi:.3f}")
            print(f"   Silhouette: {silhouette:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    return results, X_scaled, true_labels

def calculate_clustering_accuracy(true_labels, cluster_labels):
    """
    Calculate accuracy by finding the best assignment of cluster labels to true labels.
    """
    from scipy.optimize import linear_sum_assignment
    
    # Handle noise points in DBSCAN
    valid_mask = cluster_labels != -1
    if not np.any(valid_mask):
        return 0.0
    
    true_valid = true_labels[valid_mask]
    cluster_valid = cluster_labels[valid_mask]
    
    # Create confusion matrix
    unique_clusters = np.unique(cluster_valid)
    unique_true = np.unique(true_valid)
    
    if len(unique_clusters) == 1 or len(unique_true) == 1:
        # Only one cluster or one true class
        return np.mean(true_valid == (cluster_valid == unique_clusters[0]))
    
    # Build cost matrix for Hungarian algorithm
    cost_matrix = np.zeros((len(unique_clusters), len(unique_true)))
    
    for i, cluster_id in enumerate(unique_clusters):
        for j, true_id in enumerate(unique_true):
            # Cost is negative accuracy (since we want to maximize accuracy)
            mask = cluster_valid == cluster_id
            if np.sum(mask) > 0:
                cost_matrix[i, j] = -np.mean(true_valid[mask] == true_id)
    
    # Find optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate accuracy with optimal assignment
    total_correct = 0
    total_samples = 0
    
    for i, j in zip(row_indices, col_indices):
        cluster_id = unique_clusters[i]
        true_id = unique_true[j]
        mask = cluster_valid == cluster_id
        total_correct += np.sum(true_valid[mask] == true_id)
        total_samples += np.sum(mask)
    
    return total_correct / total_samples if total_samples > 0 else 0.0

def create_psychological_clustering_visualizations(representations, results, true_labels, trial_info):
    """
    Create comprehensive visualizations of clustering results for psychological data.
    """
    print("\n📊 Creating psychological clustering visualizations...")
    
    # Dimensionality reduction for visualization
    print("   Computing PCA and t-SNE...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(representations)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(representations)
    
    # Create comprehensive visualization
    n_algorithms = len(results)
    fig, axes = plt.subplots(3, max(n_algorithms + 1, 4), figsize=(4 * max(n_algorithms + 1, 4), 12))
    
    if n_algorithms == 0:
        print("   ⚠️ No clustering results to visualize")
        return
    
    # Ensure axes is 2D
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)
    
    # Color maps
    true_colors = ['red' if label == 1 else 'blue' for label in true_labels]
    
    # Row 1: PCA visualizations
    # True labels
    axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=true_colors, alpha=0.6, s=20)
    axes[0, 0].set_title('True Psychological States (PCA)', fontweight='bold')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    
    # Clustering results
    for i, (name, result) in enumerate(results.items()):
        if i >= axes.shape[1] - 1:
            break
            
        cluster_labels = result['cluster_labels']
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        for j, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
            axes[0, i + 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                                 c=[colors[j]], alpha=0.6, s=20, label=label)
        
        axes[0, i + 1].set_title(f'{name} (PCA)\nAcc: {result["accuracy"]:.3f}', fontweight='bold')
        axes[0, i + 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        if len(unique_clusters) <= 3:
            axes[0, i + 1].legend(fontsize=8)
    
    # Hide extra subplots in row 1
    for i in range(len(results) + 1, axes.shape[1]):
        axes[0, i].axis('off')
    
    # Row 2: t-SNE visualizations
    # True labels
    axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_colors, alpha=0.6, s=20)
    axes[1, 0].set_title('True Psychological States (t-SNE)', fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    
    # Clustering results
    for i, (name, result) in enumerate(results.items()):
        if i >= axes.shape[1] - 1:
            break
            
        cluster_labels = result['cluster_labels']
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        for j, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
            axes[1, i + 1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                                 c=[colors[j]], alpha=0.6, s=20, label=label)
        
        axes[1, i + 1].set_title(f'{name} (t-SNE)\nARI: {result["ari"]:.3f}', fontweight='bold')
        axes[1, i + 1].set_xlabel('t-SNE 1')
        if len(unique_clusters) <= 3:
            axes[1, i + 1].legend(fontsize=8)
    
    # Hide extra subplots in row 2
    for i in range(len(results) + 1, axes.shape[1]):
        axes[1, i].axis('off')
    
    # Row 3: Metrics comparison and analysis
    if results:
        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                'Algorithm': name,
                'Accuracy': result['accuracy'],
                'ARI': result['ari'],
                'NMI': result['nmi'],
                'Silhouette': result['silhouette']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Bar plot of metrics
        ax_metrics = axes[2, 0]
        x_pos = np.arange(len(metrics_data))
        width = 0.2
        
        ax_metrics.bar(x_pos - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
        ax_metrics.bar(x_pos - 0.5*width, metrics_df['ARI'], width, label='ARI', alpha=0.8)
        ax_metrics.bar(x_pos + 0.5*width, metrics_df['NMI'], width, label='NMI', alpha=0.8)
        ax_metrics.bar(x_pos + 1.5*width, metrics_df['Silhouette'], width, label='Silhouette', alpha=0.8)
        
        ax_metrics.set_xlabel('Clustering Algorithm')
        ax_metrics.set_ylabel('Score')
        ax_metrics.set_title('Clustering Metrics Comparison', fontweight='bold')
        ax_metrics.set_xticks(x_pos)
        ax_metrics.set_xticklabels([name.split(' ')[0] for name in metrics_df['Algorithm']], rotation=45)
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
        
        # Accuracy vs choice accuracy scatter plot
        if len(trial_info) > 0:
            ax_scatter = axes[2, 1]
            
            # Get choice accuracy for each trial
            choice_accuracy = trial_info['choice_correct'].values if 'choice_correct' in trial_info.columns else None
            
            if choice_accuracy is not None:
                # Color by true psychological state
                ax_scatter.scatter(choice_accuracy, true_labels, c=true_colors, alpha=0.6, s=20)
                ax_scatter.set_xlabel('Choice Correct (0/1)')
                ax_scatter.set_ylabel('True State (0=Disengaged, 1=Engaged)')
                ax_scatter.set_title('Choice Accuracy vs True State', fontweight='bold')
                ax_scatter.grid(True, alpha=0.3)
        
        # Hide remaining subplots in row 3
        for i in range(2, axes.shape[1]):
            axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'psychological_clustering_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Clustering visualizations saved as '{filename}'")
    
    plt.show()
    
    return X_pca, X_tsne

def analyze_psychological_patterns(cluster_labels, true_states, trial_info):
    """
    Analyze how discovered clusters relate to psychological patterns in the data.
    """
    print("\n🧠 Analyzing Psychological Patterns")
    print("=" * 50)
    
    # Convert to binary labels
    true_binary = (true_states == 'engaged').astype(int)
    
    # Handle DBSCAN noise points (-1 labels)
    valid_mask = cluster_labels != -1
    n_noise = np.sum(~valid_mask)
    n_valid = np.sum(valid_mask)
    
    print(f"📊 Clustering Overview:")
    print(f"   Total trials: {len(cluster_labels)}")
    print(f"   Valid clusters: {n_valid}")
    print(f"   Noise points: {n_noise}")
    print(f"   Unique clusters: {len(np.unique(cluster_labels[valid_mask])) if n_valid > 0 else 0}")
    
    if n_valid == 0:
        print("   ⚠️ No valid clusters found - all points classified as noise")
        return cluster_labels
    
    # For DBSCAN with many clusters, try to find the two largest clusters
    if len(np.unique(cluster_labels[valid_mask])) > 2:
        print(f"   📊 Multiple clusters found, analyzing two largest...")
        
        # Find the two largest clusters
        unique_clusters, counts = np.unique(cluster_labels[valid_mask], return_counts=True)
        largest_two_idx = np.argsort(counts)[-2:]
        largest_two_clusters = unique_clusters[largest_two_idx]
        
        print(f"   Largest clusters: {largest_two_clusters} with sizes {counts[largest_two_idx]}")
        
        # Create binary labels based on largest two clusters
        binary_cluster_labels = np.full_like(cluster_labels, -1)
        binary_cluster_labels[cluster_labels == largest_two_clusters[0]] = 0
        binary_cluster_labels[cluster_labels == largest_two_clusters[1]] = 1
        
        # Only analyze trials in the two largest clusters
        two_cluster_mask = (cluster_labels == largest_two_clusters[0]) | (cluster_labels == largest_two_clusters[1])
        
        if np.sum(two_cluster_mask) < 10:
            print("   ⚠️ Too few trials in largest clusters for meaningful analysis")
            return cluster_labels
        
        cluster_labels_analysis = binary_cluster_labels[two_cluster_mask]
        true_binary_analysis = true_binary[two_cluster_mask]
        trial_info_analysis = trial_info.loc[two_cluster_mask]
        
    else:
        # Standard binary clustering
        cluster_labels_analysis = cluster_labels[valid_mask]
        true_binary_analysis = true_binary[valid_mask]
        trial_info_analysis = trial_info.loc[valid_mask]
    
    # Find best cluster assignment (try both assignments)
    if len(np.unique(cluster_labels_analysis)) >= 2:
        acc1 = np.mean(true_binary_analysis == cluster_labels_analysis)
        acc2 = np.mean(true_binary_analysis == (1 - cluster_labels_analysis))
        
        if acc2 > acc1:
            cluster_labels_analysis = 1 - cluster_labels_analysis
        
        # Analyze patterns
        print(f"📊 Psychological Pattern Analysis (valid clusters only):")
        print(f"   True engaged trials: {np.sum(true_binary_analysis)} / {len(true_binary_analysis)} ({np.mean(true_binary_analysis):.1%})")
        print(f"   Predicted engaged trials: {np.sum(cluster_labels_analysis)} / {len(cluster_labels_analysis)} ({np.mean(cluster_labels_analysis):.1%})")
        
        # Analyze choice accuracy by discovered clusters
        if 'choice_correct' in trial_info_analysis.columns:
            engaged_mask = cluster_labels_analysis == 1
            disengaged_mask = cluster_labels_analysis == 0
            
            if np.sum(engaged_mask) > 0 and np.sum(disengaged_mask) > 0:
                engaged_accuracy = trial_info_analysis.loc[engaged_mask, 'choice_correct'].mean()
                disengaged_accuracy = trial_info_analysis.loc[disengaged_mask, 'choice_correct'].mean()
                
                print(f"\n🎯 Choice Accuracy by Discovered Clusters:")
                print(f"   Discovered 'Engaged' cluster accuracy: {engaged_accuracy:.3f}")
                print(f"   Discovered 'Disengaged' cluster accuracy: {disengaged_accuracy:.3f}")
                print(f"   Accuracy difference: {engaged_accuracy - disengaged_accuracy:.3f}")
        
        # Analyze response times by discovered clusters
        if 'rt' in trial_info_analysis.columns:
            engaged_mask = cluster_labels_analysis == 1
            disengaged_mask = cluster_labels_analysis == 0
            
            if np.sum(engaged_mask) > 0 and np.sum(disengaged_mask) > 0:
                engaged_rt = trial_info_analysis.loc[engaged_mask, 'rt'].mean()
                disengaged_rt = trial_info_analysis.loc[disengaged_mask, 'rt'].mean()
                
                print(f"\n⏱️ Response Times by Discovered Clusters:")
                print(f"   Discovered 'Engaged' cluster RT: {engaged_rt:.3f}s")
                print(f"   Discovered 'Disengaged' cluster RT: {disengaged_rt:.3f}s")
                print(f"   RT difference: {disengaged_rt - engaged_rt:.3f}s")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        try:
            cm = confusion_matrix(true_binary_analysis, cluster_labels_analysis)
            print(f"\n📋 Confusion Matrix:")
            print(f"   True\\Pred  Disengaged  Engaged")
            print(f"   Disengaged    {cm[0,0]:6d}     {cm[0,1]:6d}")
            print(f"   Engaged       {cm[1,0]:6d}     {cm[1,1]:6d}")
            
            # Classification report
            print(f"\n📈 Classification Report:")
            report = classification_report(true_binary_analysis, cluster_labels_analysis, 
                                         target_names=['Disengaged', 'Engaged'],
                                         zero_division=0)
            print(report)
        except Exception as e:
            print(f"   ⚠️ Could not generate classification report: {e}")
        
        return cluster_labels_analysis
    else:
        print("   ⚠️ Only one cluster found - cannot analyze patterns")
        return cluster_labels

def extract_representations(state, trainer, behavioral_features, environmental_features, batch_size=8):
    """
    Extract learned representations from the trained model.
    """
    import jax.numpy as jnp
    
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

def create_checkpoint_visualization(representations, clustering_results, true_labels, trial_info, 
                                  training_history, epoch, output_dir):
    """
    Create comprehensive visualization for a specific checkpoint.
    """
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(representations)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(representations)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # Color maps
    true_colors = ['red' if label == 1 else 'blue' for label in true_labels]
    
    # Row 1: Training progress
    ax_loss = fig.add_subplot(gs[0, :2])
    ax_loss.set_title(f'Training Progress (Epoch {epoch})', fontsize=14, fontweight='bold')
    
    epochs_range = range(1, len(training_history['total_loss']) + 1)
    ax_loss.plot(epochs_range, training_history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax_loss.plot(epochs_range, training_history['behavioral_loss'], 'r--', alpha=0.7, label='Behavioral Loss')
    ax_loss.plot(epochs_range, training_history['contrastive_loss'], 'g:', alpha=0.7, label='Contrastive Loss')
    ax_loss.plot(epochs_range, training_history['diversity_loss'], 'm-.', alpha=0.7, label='Diversity Loss')
    
    ax_loss.axvline(x=epoch, color='black', linestyle='--', alpha=0.5, label=f'Current Epoch')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Clustering metrics summary
    ax_metrics = fig.add_subplot(gs[0, 2:])
    ax_metrics.set_title(f'Clustering Performance (Epoch {epoch})', fontsize=14, fontweight='bold')
    
    if clustering_results:
        metrics_data = []
        for name, result in clustering_results.items():
            metrics_data.append({
                'Algorithm': name.split(' ')[0],  # Shortened names
                'Accuracy': result['accuracy'],
                'ARI': result['ari'],
                'NMI': result['nmi'],
                'Silhouette': result['silhouette']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        x_pos = np.arange(len(metrics_data))
        width = 0.2
        
        ax_metrics.bar(x_pos - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
        ax_metrics.bar(x_pos - 0.5*width, metrics_df['ARI'], width, label='ARI', alpha=0.8)
        ax_metrics.bar(x_pos + 0.5*width, metrics_df['NMI'], width, label='NMI', alpha=0.8)
        ax_metrics.bar(x_pos + 1.5*width, metrics_df['Silhouette'], width, label='Silhouette', alpha=0.8)
        
        ax_metrics.set_xlabel('Clustering Algorithm')
        ax_metrics.set_ylabel('Score')
        ax_metrics.set_xticks(x_pos)
        ax_metrics.set_xticklabels(metrics_df['Algorithm'], rotation=45)
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
    
    # Row 2: PCA visualizations
    # True labels
    ax_pca_true = fig.add_subplot(gs[1, 0])
    ax_pca_true.scatter(X_pca[:, 0], X_pca[:, 1], c=true_colors, alpha=0.6, s=20)
    ax_pca_true.set_title('True States (PCA)', fontweight='bold')
    ax_pca_true.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax_pca_true.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    
    # Best clustering result
    if clustering_results:
        best_algorithm = max(clustering_results.keys(), key=lambda k: clustering_results[k]['accuracy'])
        best_result = clustering_results[best_algorithm]
        
        ax_pca_best = fig.add_subplot(gs[1, 1])
        cluster_labels = best_result['cluster_labels']
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        for j, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
            ax_pca_best.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                               c=[colors[j]], alpha=0.6, s=20, label=label)
        
        ax_pca_best.set_title(f'{best_algorithm.split(" ")[0]} (PCA)\nAcc: {best_result["accuracy"]:.3f}', fontweight='bold')
        ax_pca_best.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        if len(unique_clusters) <= 3:
            ax_pca_best.legend(fontsize=8)
    
    # Choice accuracy vs representations
    ax_choice = fig.add_subplot(gs[1, 2])
    choice_accuracy = trial_info['choice_correct'].values
    ax_choice.scatter(choice_accuracy, true_labels, c=true_colors, alpha=0.6, s=20)
    ax_choice.set_xlabel('Choice Correct (0/1)')
    ax_choice.set_ylabel('True State (0=Disengaged, 1=Engaged)')
    ax_choice.set_title('Choice Accuracy vs True State', fontweight='bold')
    ax_choice.grid(True, alpha=0.3)
    
    # Response time distribution
    ax_rt = fig.add_subplot(gs[1, 3])
    engaged_rt = trial_info[true_labels == 1]['rt']
    disengaged_rt = trial_info[true_labels == 0]['rt']
    
    ax_rt.hist(engaged_rt, bins=30, alpha=0.7, label='Engaged', color='red', density=True)
    ax_rt.hist(disengaged_rt, bins=30, alpha=0.7, label='Disengaged', color='blue', density=True)
    ax_rt.set_xlabel('Response Time (s)')
    ax_rt.set_ylabel('Density')
    ax_rt.set_title('Response Time Distribution', fontweight='bold')
    ax_rt.legend()
    ax_rt.grid(True, alpha=0.3)
    
    # Row 3: t-SNE visualizations
    # True labels
    ax_tsne_true = fig.add_subplot(gs[2, 0])
    ax_tsne_true.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_colors, alpha=0.6, s=20)
    ax_tsne_true.set_title('True States (t-SNE)', fontweight='bold')
    ax_tsne_true.set_xlabel('t-SNE 1')
    ax_tsne_true.set_ylabel('t-SNE 2')
    
    # Best clustering result
    if clustering_results:
        ax_tsne_best = fig.add_subplot(gs[2, 1])
        cluster_labels = best_result['cluster_labels']
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        for j, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
            ax_tsne_best.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                                c=[colors[j]], alpha=0.6, s=20, label=label)
        
        ax_tsne_best.set_title(f'{best_algorithm.split(" ")[0]} (t-SNE)\nARI: {best_result["ari"]:.3f}', fontweight='bold')
        ax_tsne_best.set_xlabel('t-SNE 1')
        if len(unique_clusters) <= 3:
            ax_tsne_best.legend(fontsize=8)
    
    # Representation quality analysis
    ax_quality = fig.add_subplot(gs[2, 2:])
    
    # Calculate separability metrics
    engaged_mask = true_labels == 1
    disengaged_mask = true_labels == 0
    
    engaged_reps = representations[engaged_mask]
    disengaged_reps = representations[disengaged_mask]
    
    engaged_mean = np.mean(engaged_reps, axis=0)
    disengaged_mean = np.mean(disengaged_reps, axis=0)
    
    euclidean_dist = np.linalg.norm(engaged_mean - disengaged_mean)
    cosine_sim = np.dot(engaged_mean, disengaged_mean) / (np.linalg.norm(engaged_mean) * np.linalg.norm(disengaged_mean))
    
    # Sample distances for efficiency
    n_sample = min(200, len(engaged_reps), len(disengaged_reps))
    engaged_sample = engaged_reps[:n_sample]
    disengaged_sample = disengaged_reps[:n_sample]
    
    within_engaged = []
    within_disengaged = []
    between_groups = []
    
    for i in range(min(50, n_sample)):
        for j in range(i+1, min(50, n_sample)):
            within_engaged.append(np.linalg.norm(engaged_sample[i] - engaged_sample[j]))
            within_disengaged.append(np.linalg.norm(disengaged_sample[i] - disengaged_sample[j]))
        
        for j in range(min(50, n_sample)):
            between_groups.append(np.linalg.norm(engaged_sample[i] - disengaged_sample[j]))
    
    separability = np.mean(between_groups) / np.mean(within_engaged + within_disengaged)
    
    # Plot separability metrics
    metrics_names = ['Euclidean\nDistance', 'Cosine\nSimilarity', 'Separability\nRatio']
    metrics_values = [euclidean_dist, cosine_sim, separability]
    
    bars = ax_quality.bar(metrics_names, metrics_values, alpha=0.7, 
                         color=['skyblue', 'lightgreen', 'orange'])
    ax_quality.set_title('Representation Quality Metrics', fontweight='bold')
    ax_quality.set_ylabel('Value')
    ax_quality.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax_quality.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Row 4: Summary statistics
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    # Create summary text
    summary_text = f"""
CHECKPOINT SUMMARY - EPOCH {epoch}

Training Metrics:
• Total Loss: {training_history['total_loss'][-1]:.4f}
• Behavioral Loss: {training_history['behavioral_loss'][-1]:.4f}
• Contrastive Loss: {training_history['contrastive_loss'][-1]:.4f}
• Diversity Loss: {training_history['diversity_loss'][-1]:.4f}

Data Statistics:
• Total Trials: {len(true_labels):,}
• Engaged Trials: {np.sum(true_labels):,} ({np.mean(true_labels):.1%})
• Disengaged Trials: {np.sum(1-true_labels):,} ({np.mean(1-true_labels):.1%})
• Overall Choice Accuracy: {trial_info['choice_correct'].mean():.3f}

Representation Quality:
• Mean Distance Between States: {euclidean_dist:.3f}
• Cosine Similarity: {cosine_sim:.3f}
• Separability Ratio: {separability:.3f}

Best Clustering Performance:
• Algorithm: {best_algorithm if clustering_results else 'N/A'}
• Accuracy: {best_result['accuracy']:.3f if clustering_results else 'N/A'}
• ARI: {best_result['ari']:.3f if clustering_results else 'N/A'}
• NMI: {best_result['nmi']:.3f if clustering_results else 'N/A'}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Choice2Vec Training Checkpoint - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch:06d}_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    print(f"   ✅ Saved visualization to {plot_path}")
    
    return X_pca, X_tsne

def load_checkpoint_analysis(checkpoint_dir):
    """
    Load and analyze a specific checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Dictionary with loaded weights, clustering results, and analysis
    """
    import pickle
    
    print(f"📂 Loading checkpoint from {checkpoint_dir}")
    
    # Load model weights
    weights_path = os.path.join(checkpoint_dir, 'model_weights.pkl')
    if os.path.exists(weights_path):
        with open(weights_path, 'rb') as f:
            weights_data = pickle.load(f)
        print(f"   ✅ Loaded model weights (epoch {weights_data.get('epoch', 'unknown')})")
    else:
        print(f"   ❌ Model weights not found at {weights_path}")
        weights_data = None
    
    # Load clustering results
    results_path = os.path.join(checkpoint_dir, 'clustering_results.pkl')
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            clustering_data = pickle.load(f)
        print(f"   ✅ Loaded clustering results")
        print(f"       - {len(clustering_data['representations_256d'])} trial representations")
        print(f"       - {len(clustering_data['clustering_results'])} clustering algorithms")
        
        # Find best clustering result
        best_algorithm = max(clustering_data['clustering_results'].keys(), 
                           key=lambda k: clustering_data['clustering_results'][k]['accuracy'])
        best_accuracy = clustering_data['clustering_results'][best_algorithm]['accuracy']
        print(f"       - Best accuracy: {best_accuracy:.3f} ({best_algorithm})")
        
    else:
        print(f"   ❌ Clustering results not found at {results_path}")
        clustering_data = None
    
    # Check for visualization
    viz_pattern = os.path.join(checkpoint_dir, '*_visualization.png')
    import glob
    viz_files = glob.glob(viz_pattern)
    if viz_files:
        print(f"   ✅ Found visualization: {os.path.basename(viz_files[0])}")
    else:
        print(f"   ⚠️ No visualization found")
    
    return {
        'weights': weights_data,
        'clustering': clustering_data,
        'checkpoint_dir': checkpoint_dir,
        'visualization_files': viz_files
    }

def compare_checkpoints(main_output_dir):
    """
    Compare performance across all checkpoints in a training run.
    
    Args:
        main_output_dir: Path to main training output directory
    """
    import glob
    import pickle
    
    print(f"📊 Comparing checkpoints in {main_output_dir}")
    
    # Find all checkpoint directories
    checkpoint_pattern = os.path.join(main_output_dir, 'checkpoint_epoch_*')
    checkpoint_dirs = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoint_dirs:
        print(f"   ❌ No checkpoints found in {main_output_dir}")
        return None
    
    print(f"   Found {len(checkpoint_dirs)} checkpoints")
    
    # Collect data from all checkpoints
    comparison_data = []
    
    for checkpoint_dir in checkpoint_dirs:
        try:
            # Extract epoch number from directory name
            epoch = int(os.path.basename(checkpoint_dir).split('_')[-1])
            
            # Load clustering results
            results_path = os.path.join(checkpoint_dir, 'clustering_results.pkl')
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Find best clustering result
                best_algorithm = max(data['clustering_results'].keys(), 
                                   key=lambda k: data['clustering_results'][k]['accuracy'])
                best_result = data['clustering_results'][best_algorithm]
                
                comparison_data.append({
                    'epoch': epoch,
                    'total_loss': data['training_metrics']['total_loss'],
                    'behavioral_loss': data['training_metrics']['behavioral_loss'],
                    'contrastive_loss': data['training_metrics']['contrastive_loss'],
                    'diversity_loss': data['training_metrics']['diversity_loss'],
                    'best_algorithm': best_algorithm,
                    'best_accuracy': best_result['accuracy'],
                    'best_ari': best_result['ari'],
                    'best_nmi': best_result['nmi'],
                    'best_silhouette': best_result['silhouette']
                })
                
        except Exception as e:
            print(f"   ⚠️ Error loading {checkpoint_dir}: {e}")
            continue
    
    if not comparison_data:
        print(f"   ❌ No valid checkpoint data found")
        return None
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('epoch')
    
    print(f"\n📈 Training Progress Summary:")
    print(f"   Epochs: {df['epoch'].min()} - {df['epoch'].max()}")
    print(f"   Total Loss: {df['total_loss'].iloc[0]:.4f} → {df['total_loss'].iloc[-1]:.4f}")
    print(f"   Best Accuracy: {df['best_accuracy'].max():.3f} (epoch {df.loc[df['best_accuracy'].idxmax(), 'epoch']})")
    print(f"   Final Accuracy: {df['best_accuracy'].iloc[-1]:.3f}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['total_loss'], 'b-', label='Total Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['behavioral_loss'], 'r--', label='Behavioral Loss')
    axes[0, 0].plot(df['epoch'], df['contrastive_loss'], 'g:', label='Contrastive Loss')
    axes[0, 0].plot(df['epoch'], df['diversity_loss'], 'm-.', label='Diversity Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Clustering accuracy
    axes[0, 1].plot(df['epoch'], df['best_accuracy'], 'o-', color='purple', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Best Clustering Accuracy')
    axes[0, 1].set_title('Clustering Performance Progress')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Multiple clustering metrics
    axes[1, 0].plot(df['epoch'], df['best_accuracy'], 'o-', label='Accuracy', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['best_ari'], 's-', label='ARI', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['best_nmi'], '^-', label='NMI', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Clustering Metrics Progress')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Algorithm performance distribution
    algorithm_counts = df['best_algorithm'].value_counts()
    axes[1, 1].pie(algorithm_counts.values, labels=algorithm_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Best Performing Algorithms')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_path = os.path.join(main_output_dir, 'training_progress_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Comparison plot saved to {comparison_plot_path}")
    
    return df

def main():
    """
    Main psychological unsupervised classification pipeline.
    """
    print("🧠 Unsupervised Psychological State Discovery")
    print("Using Choice2Vec with Psychological Data (Correct/Incorrect Choices)")
    print("=" * 90)
    
    # Train extended InfoNCE model on psychological data
    state, data, training_history, trainer = train_extended_infonce_psychological()
    
    if state is None:
        print("❌ Training failed - cannot proceed with clustering")
        return
    
    # Load original data for analysis
    df = pd.read_csv('psychological_behavioral_data.csv')
    
    # Extract trial-level representations
    trial_representations, true_states = extract_trial_representations_psychological(state, trainer, df)
    
    # Create trial info for analysis
    trial_info = df[['trial_in_subtask', 'subtask', 'choice_correct', 'rt']].copy()
    
    # Perform unsupervised clustering
    clustering_results, X_scaled, true_labels = perform_unsupervised_clustering_psychological(
        trial_representations, true_states, trial_info
    )
    
    if not clustering_results:
        print("❌ No successful clustering results")
        return
    
    # Create visualizations
    X_pca, X_tsne = create_psychological_clustering_visualizations(
        X_scaled, clustering_results, true_labels, trial_info
    )
    
    # Analyze best clustering result
    best_algorithm = max(clustering_results.keys(), 
                        key=lambda k: clustering_results[k]['accuracy'])
    best_result = clustering_results[best_algorithm]
    
    print(f"\n🏆 BEST CLUSTERING RESULT: {best_algorithm}")
    print(f"=" * 70)
    print(f"   Accuracy: {best_result['accuracy']:.3f}")
    print(f"   ARI: {best_result['ari']:.3f}")
    print(f"   NMI: {best_result['nmi']:.3f}")
    print(f"   Silhouette Score: {best_result['silhouette']:.3f}")
    
    # Detailed analysis of best result
    best_clusters = analyze_psychological_patterns(
        best_result['cluster_labels'], true_states, trial_info
    )
    
    # Summary
    print(f"\n🎉 PSYCHOLOGICAL UNSUPERVISED DISCOVERY SUMMARY")
    print(f"=" * 90)
    print(f"✅ Successfully trained Choice2Vec with InfoNCE on psychological data")
    print(f"✅ Used correct/incorrect choices (not left/right)")
    print(f"✅ Removed value_difference (no trivial solution)")
    print(f"✅ Extracted representations for {len(trial_representations)} trials")
    print(f"✅ Tested {len(clustering_results)} clustering algorithms")
    print(f"🏆 Best unsupervised accuracy: {best_result['accuracy']:.1%} ({best_algorithm})")
    
    if best_result['accuracy'] > 0.8:
        print(f"🌟 EXCELLENT: >80% unsupervised accuracy shows strong psychological representation learning!")
    elif best_result['accuracy'] > 0.7:
        print(f"📈 GOOD: >70% unsupervised accuracy shows meaningful psychological patterns captured!")
    elif best_result['accuracy'] > 0.6:
        print(f"📊 MODERATE: >60% unsupervised accuracy shows some psychological signal captured!")
    else:
        print(f"🤔 WEAK: <60% accuracy suggests psychological representations may need improvement")
    
    print(f"\n🎯 This demonstrates whether Choice2Vec can learn meaningful psychological patterns")
    print(f"   from choice accuracy and response time patterns alone!")

if __name__ == "__main__":
    main() 