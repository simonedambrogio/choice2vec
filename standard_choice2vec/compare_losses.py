#!/usr/bin/env python3
"""
Compare different contrastive loss implementations for Choice2Vec.

This script demonstrates the differences between:
1. Simple MSE loss (baseline)
2. Cosine similarity loss 
3. wav2vec 2.0 InfoNCE loss (with negative sampling)

Based on the original wav2vec 2.0 paper and implementation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_choice2vec import train_choice2vec, extract_representations, evaluate_state_classification
from choice2vec_model import prepare_behavioral_data

def compare_loss_functions():
    """
    Compare different contrastive loss functions on the same dataset.
    """
    print("🔬 Comparing Contrastive Loss Functions")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists('behavioral_data.csv'):
        print("❌ Error: 'behavioral_data.csv' not found!")
        print("   Please run 'python generate_data.py' first to create the dataset.")
        return
    
    # Load data once for all experiments
    print("📊 Loading behavioral data...")
    df = pd.read_csv('behavioral_data.csv')
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=100, stride=20
    )
    print(f"   Dataset: {len(df)} trials, {len(behavioral_features)} windows")
    
    # Training configurations
    configs = [
        {
            'name': 'Simple MSE Loss',
            'use_cosine_loss': False,
            'use_wav2vec2_loss': False,
            'description': 'Basic MSE between projected and quantized features'
        },
        {
            'name': 'Cosine Similarity Loss', 
            'use_cosine_loss': True,
            'use_wav2vec2_loss': False,
            'description': 'Maximizes cosine similarity (your current approach)'
        },
        {
            'name': 'wav2vec 2.0 InfoNCE Loss',
            'use_cosine_loss': False,
            'use_wav2vec2_loss': True,
            'temperature': 0.1,
            'num_negatives': 100,
            'description': 'InfoNCE with negative sampling (original wav2vec 2.0)'
        }
    ]
    
    results = {}
    
    # Train with each loss function
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"🏋️ Experiment {i+1}/3: {config['name']}")
        print(f"   {config['description']}")
        print(f"{'='*60}")
        
        # Train model
        state, data, training_history, trainer = train_choice2vec(
            num_epochs=500,  # Shorter training for comparison
            batch_size=16,
            learning_rate=1e-4,
            use_cosine_loss=config.get('use_cosine_loss', False),
            use_wav2vec2_loss=config.get('use_wav2vec2_loss', False),
            temperature=config.get('temperature', 0.1),
            num_negatives=config.get('num_negatives', 100),
            verbose=True
        )
        
        if state is None:
            print(f"❌ Training failed for {config['name']}")
            continue
        
        # Extract representations and evaluate
        print(f"\n🧠 Extracting representations for {config['name']}...")
        contextualized_features, quantized_indices = extract_representations(
            state, trainer, behavioral_features, environmental_features
        )
        
        print(f"🎯 Evaluating classification performance...")
        classification_results = evaluate_state_classification(
            contextualized_features, states, test_size=0.3, verbose=True
        )
        
        # Store results
        results[config['name']] = {
            'training_history': training_history,
            'classification_accuracy': classification_results[0],
            'contextualized_features': contextualized_features,
            'final_losses': {
                'total': training_history['total_loss'][-1],
                'behavioral': training_history['behavioral_loss'][-1], 
                'contrastive': training_history['contrastive_loss'][-1],
                'diversity': training_history['diversity_loss'][-1]
            }
        }
        
        print(f"✅ {config['name']} completed!")
        print(f"   Final accuracy: {classification_results[0]:.3f}")
        print(f"   Final contrastive loss: {training_history['contrastive_loss'][-1]:.3f}")
    
    # Create comparison visualization
    print(f"\n📊 Creating comparison visualization...")
    create_comparison_plots(results, configs)
    
    # Print summary
    print_comparison_summary(results)
    
    return results

def create_comparison_plots(results, configs):
    """
    Create comprehensive comparison plots.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training curves comparison
    ax1 = plt.subplot(2, 4, 1)
    for name, result in results.items():
        epochs = range(len(result['training_history']['total_loss']))
        plt.plot(epochs, result['training_history']['total_loss'], 
                label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss Convergence', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Contrastive loss comparison
    ax2 = plt.subplot(2, 4, 2)
    for name, result in results.items():
        epochs = range(len(result['training_history']['contrastive_loss']))
        plt.plot(epochs, result['training_history']['contrastive_loss'], 
                label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Contrastive Loss Convergence', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Behavioral loss comparison
    ax3 = plt.subplot(2, 4, 3)
    for name, result in results.items():
        epochs = range(len(result['training_history']['behavioral_loss']))
        plt.plot(epochs, result['training_history']['behavioral_loss'], 
                label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Behavioral Loss')
    plt.title('Behavioral Loss Convergence', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Classification accuracy comparison
    ax4 = plt.subplot(2, 4, 4)
    names = list(results.keys())
    accuracies = [results[name]['classification_accuracy'] for name in names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(range(len(names)), accuracies, color=colors, alpha=0.8)
    plt.xlabel('Loss Function')
    plt.ylabel('Classification Accuracy')
    plt.title('Final Classification Performance', fontweight='bold')
    plt.xticks(range(len(names)), [name.replace(' Loss', '') for name in names], 
               rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, max(accuracies) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5-8. Final loss components comparison
    loss_components = ['total', 'behavioral', 'contrastive', 'diversity']
    for i, component in enumerate(loss_components):
        ax = plt.subplot(2, 4, 5 + i)
        
        values = [results[name]['final_losses'][component] for name in names]
        bars = plt.bar(range(len(names)), values, color=colors, alpha=0.8)
        
        plt.xlabel('Loss Function')
        plt.ylabel(f'{component.title()} Loss')
        plt.title(f'Final {component.title()} Loss', fontweight='bold')
        plt.xticks(range(len(names)), [name.replace(' Loss', '') for name in names],
                   rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'loss_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Comparison plots saved as '{filename}'")
    
    plt.show()

def print_comparison_summary(results):
    """
    Print a detailed comparison summary.
    """
    print(f"\n🎯 COMPARISON SUMMARY")
    print(f"=" * 80)
    
    # Sort by classification accuracy
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['classification_accuracy'], 
                          reverse=True)
    
    print(f"{'Rank':<4} {'Loss Function':<25} {'Accuracy':<10} {'Contrastive':<12} {'Behavioral':<12}")
    print(f"{'-'*4} {'-'*25} {'-'*10} {'-'*12} {'-'*12}")
    
    for rank, (name, result) in enumerate(sorted_results, 1):
        acc = result['classification_accuracy']
        cont_loss = result['final_losses']['contrastive']
        behav_loss = result['final_losses']['behavioral']
        
        print(f"{rank:<4} {name:<25} {acc:<10.3f} {cont_loss:<12.3f} {behav_loss:<12.3f}")
    
    # Analysis
    best_name, best_result = sorted_results[0]
    print(f"\n🏆 BEST PERFORMING: {best_name}")
    print(f"   Classification Accuracy: {best_result['classification_accuracy']:.3f}")
    print(f"   Final Contrastive Loss: {best_result['final_losses']['contrastive']:.3f}")
    
    print(f"\n📋 KEY INSIGHTS:")
    
    # Compare wav2vec2 vs others
    wav2vec_result = results.get('wav2vec 2.0 InfoNCE Loss')
    cosine_result = results.get('Cosine Similarity Loss')
    mse_result = results.get('Simple MSE Loss')
    
    if wav2vec_result and cosine_result:
        wav2vec_acc = wav2vec_result['classification_accuracy']
        cosine_acc = cosine_result['classification_accuracy']
        improvement = (wav2vec_acc - cosine_acc) / cosine_acc * 100
        
        if improvement > 5:
            print(f"   ✅ wav2vec 2.0 InfoNCE shows {improvement:.1f}% improvement over cosine similarity")
        elif improvement > 0:
            print(f"   📈 wav2vec 2.0 InfoNCE shows modest {improvement:.1f}% improvement")
        else:
            print(f"   📊 Cosine similarity performs {-improvement:.1f}% better than wav2vec 2.0")
    
    if wav2vec_result:
        wav2vec_cont = wav2vec_result['final_losses']['contrastive']
        print(f"   🎯 wav2vec 2.0 contrastive loss: {wav2vec_cont:.3f}")
        print(f"      (Lower is better for InfoNCE - indicates better positive/negative separation)")
    
    if cosine_result:
        cosine_cont = cosine_result['final_losses']['contrastive']
        print(f"   🎯 Cosine similarity loss: {cosine_cont:.3f}")
        print(f"      (More negative = better cosine similarity)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if wav2vec_result and wav2vec_result['classification_accuracy'] == max(r['classification_accuracy'] for r in results.values()):
        print(f"   🌟 Use wav2vec 2.0 InfoNCE loss for best performance")
        print(f"   🔧 The negative sampling provides richer learning signal")
        print(f"   📚 This matches the original wav2vec 2.0 paper findings")
    else:
        print(f"   🤔 Results suggest further hyperparameter tuning may be needed")
        print(f"   🔧 Consider adjusting temperature or number of negatives")
        print(f"   📊 Simple approaches may work well for this behavioral data")

def main():
    """
    Main comparison pipeline.
    """
    print("🔬 Choice2Vec Loss Function Comparison")
    print("Based on wav2vec 2.0 paper and implementation")
    print("=" * 70)
    
    results = compare_loss_functions()
    
    if results:
        print(f"\n🎉 Comparison Complete!")
        print(f"   All loss functions tested on the same dataset")
        print(f"   Results show the impact of different contrastive learning approaches")
        print(f"   wav2vec 2.0 InfoNCE implements the original paper's methodology")

if __name__ == "__main__":
    main() 