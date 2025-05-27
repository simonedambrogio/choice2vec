#!/usr/bin/env python3
"""
Test convergence speed of wav2vec 2.0 InfoNCE loss with longer training.

This script specifically tests whether the InfoNCE loss just needs more epochs
to achieve competitive performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_choice2vec import train_choice2vec, extract_representations, evaluate_state_classification
from choice2vec_model import prepare_behavioral_data

def test_infonce_convergence():
    """
    Test wav2vec 2.0 InfoNCE loss with extended training.
    """
    print("🔬 Testing wav2vec 2.0 InfoNCE Loss Convergence")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists('behavioral_data.csv'):
        print("❌ Error: 'behavioral_data.csv' not found!")
        print("   Please run 'python generate_data.py' first to create the dataset.")
        return
    
    # Load data
    print("📊 Loading behavioral data...")
    df = pd.read_csv('behavioral_data.csv')
    behavioral_features, environmental_features, states = prepare_behavioral_data(
        df, window_size=100, stride=20
    )
    print(f"   Dataset: {len(df)} trials, {len(behavioral_features)} windows")
    
    # Test different epoch counts
    epoch_counts = [500, 1000, 2000]
    results = {}
    
    for epochs in epoch_counts:
        print(f"\n{'='*60}")
        print(f"🏋️ Training wav2vec 2.0 InfoNCE for {epochs} epochs")
        print(f"{'='*60}")
        
        # Train model
        state, data, training_history, trainer = train_choice2vec(
            num_epochs=epochs,
            batch_size=16,
            learning_rate=1e-4,
            use_cosine_loss=False,
            use_wav2vec2_loss=True,
            temperature=0.1,
            num_negatives=100,
            verbose=True
        )
        
        if state is None:
            print(f"❌ Training failed for {epochs} epochs")
            continue
        
        # Extract representations and evaluate
        print(f"\n🧠 Extracting representations...")
        contextualized_features, quantized_indices = extract_representations(
            state, trainer, behavioral_features, environmental_features
        )
        
        print(f"🎯 Evaluating classification performance...")
        classification_results = evaluate_state_classification(
            contextualized_features, states, test_size=0.3, verbose=True
        )
        
        # Store results
        results[epochs] = {
            'training_history': training_history,
            'classification_accuracy': classification_results[0],
            'final_losses': {
                'total': training_history['total_loss'][-1],
                'behavioral': training_history['behavioral_loss'][-1], 
                'contrastive': training_history['contrastive_loss'][-1],
                'diversity': training_history['diversity_loss'][-1]
            }
        }
        
        print(f"✅ {epochs} epochs completed!")
        print(f"   Final accuracy: {classification_results[0]:.3f}")
        print(f"   Final contrastive loss: {training_history['contrastive_loss'][-1]:.3f}")
        print(f"   Final behavioral loss: {training_history['behavioral_loss'][-1]:.3f}")
    
    # Create convergence analysis plots
    print(f"\n📊 Creating convergence analysis...")
    create_convergence_plots(results)
    
    # Print convergence summary
    print_convergence_summary(results)
    
    return results

def create_convergence_plots(results):
    """
    Create plots showing convergence over different epoch counts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Contrastive loss convergence
    ax1 = axes[0, 0]
    for epochs, result in results.items():
        history = result['training_history']
        ax1.plot(history['contrastive_loss'], label=f'{epochs} epochs', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Contrastive Loss')
    ax1.set_title('InfoNCE Contrastive Loss Convergence', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Behavioral loss convergence
    ax2 = axes[0, 1]
    for epochs, result in results.items():
        history = result['training_history']
        ax2.plot(history['behavioral_loss'], label=f'{epochs} epochs', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Behavioral Loss')
    ax2.set_title('Behavioral Loss Convergence', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Total loss convergence
    ax3 = axes[1, 0]
    for epochs, result in results.items():
        history = result['training_history']
        ax3.plot(history['total_loss'], label=f'{epochs} epochs', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Loss')
    ax3.set_title('Total Loss Convergence', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Classification accuracy vs epochs
    ax4 = axes[1, 1]
    epoch_counts = list(results.keys())
    accuracies = [results[epochs]['classification_accuracy'] for epochs in epoch_counts]
    
    ax4.plot(epoch_counts, accuracies, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Training Epochs')
    ax4.set_ylabel('Classification Accuracy')
    ax4.set_title('Accuracy vs Training Duration', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(epoch_counts, accuracies):
        ax4.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'infonce_convergence_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Convergence plots saved as '{filename}'")
    
    plt.show()

def print_convergence_summary(results):
    """
    Print detailed convergence analysis.
    """
    print(f"\n🎯 CONVERGENCE ANALYSIS SUMMARY")
    print(f"=" * 80)
    
    print(f"{'Epochs':<8} {'Accuracy':<10} {'Contrastive':<12} {'Behavioral':<12} {'Total':<10}")
    print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    
    for epochs in sorted(results.keys()):
        result = results[epochs]
        acc = result['classification_accuracy']
        cont_loss = result['final_losses']['contrastive']
        behav_loss = result['final_losses']['behavioral']
        total_loss = result['final_losses']['total']
        
        print(f"{epochs:<8} {acc:<10.3f} {cont_loss:<12.3f} {behav_loss:<12.3f} {total_loss:<10.3f}")
    
    # Analysis
    epoch_counts = sorted(results.keys())
    accuracies = [results[epochs]['classification_accuracy'] for epochs in epoch_counts]
    contrastive_losses = [results[epochs]['final_losses']['contrastive'] for epochs in epoch_counts]
    
    print(f"\n📋 KEY INSIGHTS:")
    
    # Check if accuracy improves with more epochs
    if len(accuracies) >= 2:
        improvement_500_to_1000 = (accuracies[1] - accuracies[0]) / accuracies[0] * 100
        if len(accuracies) >= 3:
            improvement_1000_to_2000 = (accuracies[2] - accuracies[1]) / accuracies[1] * 100
            
            print(f"   📈 Accuracy improvement 500→1000 epochs: {improvement_500_to_1000:+.1f}%")
            print(f"   📈 Accuracy improvement 1000→2000 epochs: {improvement_1000_to_2000:+.1f}%")
            
            if improvement_500_to_1000 > 5:
                print(f"   ✅ Significant improvement with longer training!")
            elif improvement_500_to_1000 > 0:
                print(f"   📊 Modest improvement with longer training")
            else:
                print(f"   🤔 No improvement - may have converged")
    
    # Check contrastive loss convergence
    if len(contrastive_losses) >= 2:
        cont_reduction = (contrastive_losses[0] - contrastive_losses[-1]) / contrastive_losses[0] * 100
        print(f"   🎯 Contrastive loss reduction: {cont_reduction:.1f}%")
        
        if contrastive_losses[-1] > 3.0:
            print(f"   ⚠️  Contrastive loss still high ({contrastive_losses[-1]:.3f}) - needs more training")
        elif contrastive_losses[-1] > 1.0:
            print(f"   📊 Contrastive loss moderate ({contrastive_losses[-1]:.3f}) - partially converged")
        else:
            print(f"   ✅ Contrastive loss low ({contrastive_losses[-1]:.3f}) - well converged")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    best_accuracy = max(accuracies)
    best_epochs = epoch_counts[accuracies.index(best_accuracy)]
    
    if best_epochs == max(epoch_counts):
        print(f"   🚀 Try even longer training (3000+ epochs)")
        print(f"   🔧 InfoNCE may need 5-10x more epochs than simple losses")
    else:
        print(f"   ✅ Optimal training duration: {best_epochs} epochs")
        print(f"   📊 Further training shows diminishing returns")
    
    # Compare to baseline results
    print(f"\n📊 COMPARISON TO BASELINE RESULTS (500 epochs):")
    print(f"   Simple MSE Loss: 83.8% accuracy")
    print(f"   Cosine Similarity: 78.4% accuracy") 
    print(f"   InfoNCE ({max(epoch_counts)} epochs): {max(accuracies):.1%} accuracy")
    
    if max(accuracies) > 0.838:
        print(f"   🌟 InfoNCE surpasses MSE with longer training!")
    elif max(accuracies) > 0.784:
        print(f"   📈 InfoNCE surpasses cosine similarity with longer training")
    else:
        print(f"   🤔 InfoNCE still underperforms - may need hyperparameter tuning")

def main():
    """
    Main convergence testing pipeline.
    """
    print("🔬 wav2vec 2.0 InfoNCE Convergence Analysis")
    print("Testing whether InfoNCE just needs more training time")
    print("=" * 70)
    
    results = test_infonce_convergence()
    
    if results:
        print(f"\n🎉 Convergence Analysis Complete!")
        print(f"   Results show the impact of training duration on InfoNCE performance")
        print(f"   This helps determine if InfoNCE is slow or fundamentally unsuited")

if __name__ == "__main__":
    main() 