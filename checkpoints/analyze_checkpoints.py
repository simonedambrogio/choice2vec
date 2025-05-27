#!/usr/bin/env python3
"""
Checkpoint Analysis Script for Choice2Vec Training

This script provides utilities to analyze saved checkpoints from the 
psychological_unsupervised_classification.py training runs.

Usage:
    python analyze_checkpoints.py --training_dir training_checkpoints_2024-01-15_10-30-45
    python analyze_checkpoints.py --checkpoint checkpoint_epoch_050000
    python analyze_checkpoints.py --compare training_checkpoints_2024-01-15_10-30-45
"""

import os
import sys
import argparse
import glob
from psychological_unsupervised_classification import (
    load_checkpoint_analysis, 
    compare_checkpoints
)

def main():
    parser = argparse.ArgumentParser(description='Analyze Choice2Vec training checkpoints')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--training_dir', type=str, 
                      help='Path to training directory (analyze all checkpoints)')
    group.add_argument('--checkpoint', type=str,
                      help='Path to specific checkpoint directory')
    group.add_argument('--compare', type=str,
                      help='Path to training directory (create comparison plots)')
    group.add_argument('--list', action='store_true',
                      help='List all available training directories')
    
    args = parser.parse_args()
    
    if args.list:
        # List all available training directories
        pattern = 'training_checkpoints_*'
        training_dirs = sorted(glob.glob(pattern))
        
        if not training_dirs:
            print("❌ No training directories found")
            print("   Looking for directories matching: training_checkpoints_*")
            return
        
        print(f"📁 Found {len(training_dirs)} training directories:")
        for i, dir_path in enumerate(training_dirs, 1):
            # Count checkpoints in each directory
            checkpoint_pattern = os.path.join(dir_path, 'checkpoint_epoch_*')
            checkpoints = glob.glob(checkpoint_pattern)
            print(f"   {i:2d}. {dir_path} ({len(checkpoints)} checkpoints)")
        
        print(f"\nUsage examples:")
        print(f"   python analyze_checkpoints.py --compare {training_dirs[-1]}")
        print(f"   python analyze_checkpoints.py --training_dir {training_dirs[-1]}")
        
        return
    
    elif args.training_dir:
        # Analyze all checkpoints in a training directory
        if not os.path.exists(args.training_dir):
            print(f"❌ Training directory not found: {args.training_dir}")
            return
        
        print(f"🔍 Analyzing training directory: {args.training_dir}")
        
        # Find all checkpoints
        checkpoint_pattern = os.path.join(args.training_dir, 'checkpoint_epoch_*')
        checkpoints = sorted(glob.glob(checkpoint_pattern))
        
        if not checkpoints:
            print(f"   ❌ No checkpoints found in {args.training_dir}")
            return
        
        print(f"   Found {len(checkpoints)} checkpoints")
        
        # Analyze each checkpoint
        for i, checkpoint_dir in enumerate(checkpoints):
            print(f"\n📊 Checkpoint {i+1}/{len(checkpoints)}: {os.path.basename(checkpoint_dir)}")
            analysis = load_checkpoint_analysis(checkpoint_dir)
            
            if analysis['clustering']:
                clustering_data = analysis['clustering']
                best_algorithm = max(clustering_data['clustering_results'].keys(), 
                                   key=lambda k: clustering_data['clustering_results'][k]['accuracy'])
                best_accuracy = clustering_data['clustering_results'][best_algorithm]['accuracy']
                
                print(f"       Best clustering: {best_accuracy:.3f} ({best_algorithm})")
                print(f"       Training loss: {clustering_data['training_metrics']['total_loss']:.4f}")
        
        # Create comparison
        print(f"\n📈 Creating comparison analysis...")
        comparison_df = compare_checkpoints(args.training_dir)
        
        if comparison_df is not None:
            print(f"\n🎯 Key Insights:")
            best_epoch = comparison_df.loc[comparison_df['best_accuracy'].idxmax(), 'epoch']
            best_accuracy = comparison_df['best_accuracy'].max()
            final_accuracy = comparison_df['best_accuracy'].iloc[-1]
            
            print(f"   • Peak performance: {best_accuracy:.3f} at epoch {best_epoch}")
            print(f"   • Final performance: {final_accuracy:.3f}")
            print(f"   • Performance trend: {'↗️ Improving' if final_accuracy > best_accuracy * 0.95 else '📉 May have peaked'}")
    
    elif args.checkpoint:
        # Analyze a specific checkpoint
        if not os.path.exists(args.checkpoint):
            print(f"❌ Checkpoint directory not found: {args.checkpoint}")
            return
        
        print(f"🔍 Analyzing specific checkpoint: {args.checkpoint}")
        analysis = load_checkpoint_analysis(args.checkpoint)
        
        if analysis['clustering']:
            clustering_data = analysis['clustering']
            
            print(f"\n📊 Detailed Analysis:")
            print(f"   Epoch: {clustering_data['epoch']}")
            print(f"   Total trials: {len(clustering_data['representations_256d'])}")
            print(f"   Representation dimension: {clustering_data['representations_256d'].shape[1]}")
            
            print(f"\n🎯 Clustering Results:")
            for name, result in clustering_data['clustering_results'].items():
                print(f"   {name:25s}: Acc={result['accuracy']:.3f}, ARI={result['ari']:.3f}, NMI={result['nmi']:.3f}")
            
            print(f"\n🏋️ Training Metrics:")
            metrics = clustering_data['training_metrics']
            print(f"   Total Loss: {metrics['total_loss']:.4f}")
            print(f"   Behavioral Loss: {metrics['behavioral_loss']:.4f}")
            print(f"   Contrastive Loss: {metrics['contrastive_loss']:.4f}")
            print(f"   Diversity Loss: {metrics['diversity_loss']:.4f}")
    
    elif args.compare:
        # Create comparison plots only
        if not os.path.exists(args.compare):
            print(f"❌ Training directory not found: {args.compare}")
            return
        
        print(f"📈 Creating comparison plots for: {args.compare}")
        comparison_df = compare_checkpoints(args.compare)
        
        if comparison_df is not None:
            print(f"✅ Comparison complete!")
        else:
            print(f"❌ Failed to create comparison")

if __name__ == "__main__":
    main() 