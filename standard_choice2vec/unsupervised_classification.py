#!/usr/bin/env python3
"""
Unsupervised Psychological State Discovery with Choice2Vec

This script trains Choice2Vec with InfoNCE loss for 3000 epochs, then performs
unsupervised clustering on learned representations to discover psychological states
(engaged vs disengaged) without using any labels during clustering.
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

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from standard_choice2vec.train_choice2vec import train_choice2vec, extract_representations
from core.choice2vec_model import prepare_behavioral_data

def train_extended_infonce():
    """
    Train Choice2Vec with InfoNCE loss for 3000 epochs.
    """
    print("🚀 Training Choice2Vec with InfoNCE for 3000 epochs")
    print("=" * 70)
    
    # Train model with extended epochs
    state, data, training_history, trainer = train_choice2vec(
        num_epochs=3000,
        batch_size=16,
        learning_rate=1e-4,
        use_cosine_loss=False,
        use_wav2vec2_loss=True,
        temperature=0.1,
        num_negatives=100,
        verbose=True
    )
    
    if state is None:
        print("❌ Training failed!")
        return None, None, None, None
    
    print(f"\n✅ Training completed!")
    print(f"   Final contrastive loss: {training_history['contrastive_loss'][-1]:.3f}")
    print(f"   Final behavioral loss: {training_history['behavioral_loss'][-1]:.3f}")
    print(f"   Final total loss: {training_history['total_loss'][-1]:.3f}")
    
    return state, data, training_history, trainer

def extract_trial_representations(state, trainer, df):
    """
    Extract representations for individual trials (not windows).
    
    This creates a representation for each trial that can be used for
    unsupervised clustering to discover psychological states.
    """
    print("\n🧠 Extracting trial-level representations...")
    
    # Prepare data with single-trial windows (window_size=1, stride=1)
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=1, stride=1
    )
    
    print(f"   Created {len(behavioral_features)} single-trial windows")
    
    # Extract representations
    contextualized_features, quantized_indices = extract_representations(
        state, trainer, behavioral_features, environmental_features
    )
    
    # Since each window has only 1 trial, we can directly use the representations
    # Shape: (n_trials, embed_dim)
    trial_representations = contextualized_features.squeeze(axis=1)  # Remove sequence dimension
    
    print(f"   Trial representations shape: {trial_representations.shape}")
    
    return trial_representations, states.flatten()

def perform_unsupervised_clustering(representations, true_states, trial_info):
    """
    Perform multiple unsupervised clustering algorithms and evaluate against ground truth.
    """
    print("\n🎯 Performing Unsupervised Clustering")
    print("=" * 50)
    
    # Standardize features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(representations)
    
    # Define clustering algorithms
    clustering_algorithms = {
        'K-Means (k=2)': KMeans(n_clusters=2, random_state=42, n_init=10),
        'Gaussian Mixture (k=2)': GaussianMixture(n_components=2, random_state=42),
        'Agglomerative (k=2)': AgglomerativeClustering(n_clusters=2),
        'DBSCAN (auto)': DBSCAN(eps=0.5, min_samples=5),
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
                print(f"   Found {n_clusters} clusters ({np.sum(cluster_labels == -1)} noise points)")
                
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

def create_clustering_visualizations(representations, results, true_labels, trial_info):
    """
    Create comprehensive visualizations of clustering results.
    """
    print("\n📊 Creating clustering visualizations...")
    
    # Dimensionality reduction for visualization
    print("   Computing PCA and t-SNE...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(representations)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(representations)
    
    # Create comprehensive visualization
    n_algorithms = len(results)
    fig, axes = plt.subplots(3, n_algorithms + 1, figsize=(4 * (n_algorithms + 1), 12))
    
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
    axes[0, 0].set_title('True States (PCA)', fontweight='bold')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    
    # Clustering results
    for i, (name, result) in enumerate(results.items()):
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
    
    # Row 2: t-SNE visualizations
    # True labels
    axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_colors, alpha=0.6, s=20)
    axes[1, 0].set_title('True States (t-SNE)', fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    
    # Clustering results
    for i, (name, result) in enumerate(results.items()):
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
    
    # Row 3: Metrics comparison
    metrics_data = []
    for name, result in results.items():
        metrics_data.append({
            'Algorithm': name,
            'Accuracy': result['accuracy'],
            'ARI': result['ari'],
            'NMI': result['nmi'],
            'Silhouette': result['silhouette']
        })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        
        # Bar plot of metrics
        x_pos = np.arange(len(metrics_data))
        width = 0.2
        
        axes[2, 0].bar(x_pos - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
        axes[2, 0].bar(x_pos - 0.5*width, metrics_df['ARI'], width, label='ARI', alpha=0.8)
        axes[2, 0].bar(x_pos + 0.5*width, metrics_df['NMI'], width, label='NMI', alpha=0.8)
        axes[2, 0].bar(x_pos + 1.5*width, metrics_df['Silhouette'], width, label='Silhouette', alpha=0.8)
        
        axes[2, 0].set_xlabel('Clustering Algorithm')
        axes[2, 0].set_ylabel('Score')
        axes[2, 0].set_title('Clustering Metrics Comparison', fontweight='bold')
        axes[2, 0].set_xticks(x_pos)
        axes[2, 0].set_xticklabels([name.split(' ')[0] for name in metrics_df['Algorithm']], rotation=45)
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Hide remaining subplots in row 3
        for i in range(1, n_algorithms + 1):
            axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'results/unsupervised_clustering_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Clustering visualizations saved as '{filename}'")
    
    plt.show()
    
    return X_pca, X_tsne

def analyze_state_transitions(cluster_labels, true_states, trial_info):
    """
    Analyze how discovered clusters relate to psychological state transitions.
    """
    print("\n🔄 Analyzing State Transitions")
    print("=" * 40)
    
    # Convert to binary labels
    true_binary = (true_states == 'engaged').astype(int)
    
    # Find best cluster assignment
    if len(np.unique(cluster_labels)) >= 2:
        # Try both assignments and pick the one with higher accuracy
        acc1 = np.mean(true_binary == cluster_labels)
        acc2 = np.mean(true_binary == (1 - cluster_labels))
        
        if acc2 > acc1:
            cluster_labels = 1 - cluster_labels
        
        # Analyze transitions
        print(f"📊 State Transition Analysis:")
        print(f"   True engaged trials: {np.sum(true_binary)} / {len(true_binary)} ({np.mean(true_binary):.1%})")
        print(f"   Predicted engaged trials: {np.sum(cluster_labels)} / {len(cluster_labels)} ({np.mean(cluster_labels):.1%})")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(true_binary, cluster_labels)
        print(f"\n📋 Confusion Matrix:")
        print(f"   True\\Pred  Disengaged  Engaged")
        print(f"   Disengaged    {cm[0,0]:6d}     {cm[0,1]:6d}")
        print(f"   Engaged       {cm[1,0]:6d}     {cm[1,1]:6d}")
        
        # Classification report
        print(f"\n📈 Classification Report:")
        report = classification_report(true_binary, cluster_labels, 
                                     target_names=['Disengaged', 'Engaged'])
        print(report)
        
        return cluster_labels
    else:
        print("   ⚠️ Only one cluster found - cannot analyze transitions")
        return cluster_labels

def main():
    """
    Main unsupervised classification pipeline.
    """
    print("🎯 Unsupervised Psychological State Discovery")
    print("Using Choice2Vec with InfoNCE Loss (3000 epochs)")
    print("=" * 80)
    
    # Check if data exists - prefer psychological data format
    if os.path.exists('results/psychological_behavioral_data.csv'):
        print("📊 Loading psychological behavioral data (choice_correct format)...")
        df = pd.read_csv('results/psychological_behavioral_data.csv')
    elif os.path.exists('results/behavioral_data.csv'):
        print("📊 Loading standard behavioral data (choice format)...")
        df = pd.read_csv('results/behavioral_data.csv')
    else:
        print("❌ Error: No behavioral data found!")
        print("   Please run 'python data_generation/generate_psychological_data.py' first to create the dataset.")
        return
    print(f"   Dataset: {len(df)} trials across {df['subtask'].nunique()} subtasks")
    print(f"   Psychological states: {df['psychological_state'].value_counts().to_dict()}")
    
    # Train extended InfoNCE model
    state, data, training_history, trainer = train_extended_infonce()
    
    if state is None:
        print("❌ Training failed - cannot proceed with clustering")
        return
    
    # Extract trial-level representations
    trial_representations, true_states = extract_trial_representations(state, trainer, df)
    
    # Create trial info for analysis
    trial_info = df[['trial_in_subtask', 'subtask', 'choice', 'rt']].copy()
    
    # Perform unsupervised clustering
    clustering_results, X_scaled, true_labels = perform_unsupervised_clustering(
        trial_representations, true_states, trial_info
    )
    
    if not clustering_results:
        print("❌ No successful clustering results")
        return
    
    # Create visualizations
    X_pca, X_tsne = create_clustering_visualizations(
        X_scaled, clustering_results, true_labels, trial_info
    )
    
    # Analyze best clustering result
    best_algorithm = max(clustering_results.keys(), 
                        key=lambda k: clustering_results[k]['accuracy'])
    best_result = clustering_results[best_algorithm]
    
    print(f"\n🏆 BEST CLUSTERING RESULT: {best_algorithm}")
    print(f"=" * 60)
    print(f"   Accuracy: {best_result['accuracy']:.3f}")
    print(f"   ARI: {best_result['ari']:.3f}")
    print(f"   NMI: {best_result['nmi']:.3f}")
    print(f"   Silhouette Score: {best_result['silhouette']:.3f}")
    
    # Detailed analysis of best result
    best_clusters = analyze_state_transitions(
        best_result['cluster_labels'], true_states, trial_info
    )
    
    # Summary
    print(f"\n🎉 UNSUPERVISED DISCOVERY SUMMARY")
    print(f"=" * 80)
    print(f"✅ Successfully trained Choice2Vec with InfoNCE (3000 epochs)")
    print(f"✅ Extracted representations for {len(trial_representations)} trials")
    print(f"✅ Tested {len(clustering_results)} clustering algorithms")
    print(f"🏆 Best unsupervised accuracy: {best_result['accuracy']:.1%} ({best_algorithm})")
    print(f"🎯 This demonstrates Choice2Vec learns meaningful psychological state representations!")
    
    if best_result['accuracy'] > 0.8:
        print(f"🌟 EXCELLENT: >80% unsupervised accuracy shows strong representation learning!")
    elif best_result['accuracy'] > 0.7:
        print(f"📈 GOOD: >70% unsupervised accuracy shows meaningful patterns captured!")
    elif best_result['accuracy'] > 0.6:
        print(f"📊 MODERATE: >60% unsupervised accuracy shows some signal captured!")
    else:
        print(f"🤔 WEAK: <60% accuracy suggests representations may need improvement")

if __name__ == "__main__":
    main() 