# Choice2Vec Training Checkpoints

This document describes the enhanced checkpoint functionality for the Choice2Vec psychological unsupervised classification training.

## Overview

The modified `psychological_unsupervised_classification.py` script now supports:
- **150,000 epoch training** with automatic checkpoints every 10,000 epochs
- **Comprehensive data saving** at each checkpoint
- **Automatic visualization generation**
- **Post-training analysis tools**

## What Gets Saved at Each Checkpoint

Every 10,000 epochs, the script creates a subdirectory with:

### 1. Model Weights (`model_weights.pkl`)
```python
saveable_state = {
    'params': state.params,                    # JAX model parameters
    'model_config': {...},                     # Architecture configuration
    'training_config': {...},                  # Training hyperparameters
    'data_config': {...},                      # Data preprocessing settings
    'training_history': training_history,      # Loss curves up to this point
    'epoch': checkpoint_epoch                  # Current epoch number
}
```

### 2. Clustering Results (`clustering_results.pkl`)
```python
results_object = {
    'epoch': checkpoint_epoch,
    'true_states': true_states,                # Original string labels ('engaged'/'disengaged')
    'true_labels_binary': true_labels,         # Binary labels (0/1)
    'representations_256d': trial_representations,  # 256D representations from model
    'clustering_results': clustering_results,  # All clustering algorithm results
    'trial_info': [...],                       # Trial metadata (RT, accuracy, etc.)
    'training_metrics': {...}                  # Current loss values
}
```

### 3. Comprehensive Visualization (`checkpoint_epoch_XXXXXX_visualization.png`)
A 4-row visualization showing:
- **Row 1**: Training progress curves + clustering performance metrics
- **Row 2**: PCA visualizations (true states + best clustering) + choice accuracy + RT distributions
- **Row 3**: t-SNE visualizations + representation quality metrics
- **Row 4**: Summary statistics and key metrics

## Directory Structure

```
training_checkpoints_2024-01-15_10-30-45/
├── checkpoint_epoch_010000/
│   ├── model_weights.pkl
│   ├── clustering_results.pkl
│   └── checkpoint_epoch_010000_visualization.png
├── checkpoint_epoch_020000/
│   ├── model_weights.pkl
│   ├── clustering_results.pkl
│   └── checkpoint_epoch_020000_visualization.png
├── ...
├── checkpoint_epoch_150000/
│   ├── model_weights.pkl
│   ├── clustering_results.pkl
│   └── checkpoint_epoch_150000_visualization.png
└── training_progress_comparison.png  # Created by analysis script
```

## Usage

### Running Training with Checkpoints

```bash
python psychological_unsupervised_classification.py
```

This will:
- Train for 150,000 epochs
- Save checkpoints every 10,000 epochs
- Create timestamped output directory
- Generate visualizations automatically

### Analyzing Checkpoints

Use the provided analysis script:

```bash
# List all available training runs
python analyze_checkpoints.py --list

# Analyze all checkpoints in a training run
python analyze_checkpoints.py --training_dir training_checkpoints_2024-01-15_10-30-45

# Analyze a specific checkpoint
python analyze_checkpoints.py --checkpoint training_checkpoints_2024-01-15_10-30-45/checkpoint_epoch_050000

# Create comparison plots only
python analyze_checkpoints.py --compare training_checkpoints_2024-01-15_10-30-45
```

## Loading Saved Models

### Loading Model Weights
```python
import pickle
import jax

# Load checkpoint
with open('checkpoint_epoch_050000/model_weights.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# Access components
params = checkpoint['params']
model_config = checkpoint['model_config']
training_history = checkpoint['training_history']
epoch = checkpoint['epoch']

# Recreate model
from choice2vec_model import Choice2Vec
model = Choice2Vec(**model_config)
```

### Loading Clustering Results
```python
# Load clustering analysis
with open('checkpoint_epoch_050000/clustering_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Access components
representations = results['representations_256d']  # Shape: (n_trials, 256)
true_states = results['true_states']               # ['engaged', 'disengaged', ...]
clustering_results = results['clustering_results'] # All algorithm results
trial_info = results['trial_info']                # Trial metadata

# Get best clustering result
best_algorithm = max(clustering_results.keys(), 
                    key=lambda k: clustering_results[k]['accuracy'])
best_result = clustering_results[best_algorithm]
print(f"Best: {best_algorithm} with {best_result['accuracy']:.3f} accuracy")
```

## Key Features

### 1. Memory Efficient
- Visualizations are saved and closed immediately to prevent memory buildup
- Only essential model components are saved (not full training state)
- Clustering is performed on-demand at checkpoints

### 2. Comprehensive Analysis
- 8 different clustering algorithms tested at each checkpoint
- Multiple evaluation metrics (Accuracy, ARI, NMI, Silhouette)
- Representation quality analysis (separability, distances)

### 3. Progress Tracking
- Training loss curves
- Clustering performance over time
- Best epoch identification
- Performance trend analysis

### 4. Flexible Analysis
- Load any specific checkpoint
- Compare across all checkpoints
- Extract representations for custom analysis
- Recreate models from saved weights

## Expected Training Time

- **150,000 epochs** ≈ 12-15 hours on GPU
- **Checkpoint overhead** ≈ 2-3 minutes per checkpoint
- **Total checkpoints**: 15 (every 10,000 epochs)
- **Storage per checkpoint**: ~50-100 MB

## Monitoring Training

The script prints progress every 1,000 epochs and detailed checkpoint information:

```
Epoch   1000/150000 | Total: 2.456 | Behavioral: 1.234 | Contrastive: -0.567 | Diversity: -0.890

💾 Saving checkpoint at epoch 10000...
   ✅ Saved model weights to .../model_weights.pkl
   🧠 Extracting representations and performing clustering...
   📊 Running K-Means (k=2)...
       Accuracy: 0.742
   📊 Running Gaussian Mixture (k=2)...
       Accuracy: 0.738
   ...
   ✅ Saved clustering results to .../clustering_results.pkl
   📊 Creating visualization...
   ✅ Saved visualization to .../checkpoint_epoch_010000_visualization.png
   ✅ Checkpoint 10000 completed in .../checkpoint_epoch_010000
```

## Troubleshooting

### Common Issues

1. **Out of memory during clustering**
   - Reduce `n_sample` in representation quality analysis
   - Use fewer clustering algorithms

2. **Slow checkpoint saving**
   - Check disk space and I/O performance
   - Consider reducing visualization complexity

3. **Missing dependencies**
   - Ensure `scipy` is installed for Hungarian algorithm
   - Ensure `scikit-learn` is installed for clustering

### Recovery from Interruption

If training is interrupted, you can:
1. Check the last saved checkpoint
2. Modify the script to resume from that epoch
3. Load the saved model weights and continue training

This checkpoint system provides comprehensive tracking of your Choice2Vec training progress and enables detailed analysis of how the model learns psychological representations over time. 