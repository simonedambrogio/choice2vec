# Choice2Vec: Self-Supervised Learning for Behavioral Data

A comprehensive framework for self-supervised learning on behavioral data, featuring multiple approaches and extensive analysis tools.

## 🏗️ Project Structure

```
Choice2Vec/
├── core/                           # Core model implementations
│   ├── choice2vec_model.py        # Base Choice2Vec model and trainer
│   └── __init__.py
├── data_generation/                # Data generation scripts
│   ├── generate_data.py           # Basic behavioral data generation
│   ├── generate_psychological_data.py  # Enhanced psychological data
│   └── __init__.py
├── standard_choice2vec/            # Standard Choice2Vec approach
│   ├── train_choice2vec.py        # Main training script
│   ├── unsupervised_classification.py  # Alternative training approach
│   ├── compare_losses.py          # Loss function comparisons
│   ├── test_convergence.py        # Convergence analysis
│   └── __init__.py
├── disentangled_choice2vec/        # Disentangled representation learning
│   ├── disentangled_choice2vec.py # Disentangled model implementation
│   ├── train_disentangled_choice2vec.py  # Training and comparison
│   ├── test_disentanglement.py    # Quick testing script
│   └── __init__.py
├── checkpoints/                    # Long-term training with checkpoints
│   ├── psychological_unsupervised_classification.py  # 150k epoch training
│   ├── analyze_checkpoints.py     # Checkpoint analysis tools
│   └── __init__.py
├── evaluation/                     # Model evaluation and analysis
│   ├── evaluate_saved_model.py    # Load and evaluate saved models
│   ├── list_models.py             # List available models
│   ├── analyze_learning_curves.py # Learning curve analysis
│   └── __init__.py
├── utils/                          # Utility functions (future)
│   └── __init__.py
├── docs/                           # Documentation
│   ├── README.md                  # Main documentation (this file)
│   ├── DISENTANGLEMENT_README.md  # Disentanglement techniques guide
│   └── CHECKPOINT_README.md       # Checkpoint system guide
├── results/                        # Generated data and plots
│   ├── *.csv                      # Generated datasets
│   ├── *.png                      # Generated visualizations
│   └── *.pkl                      # Saved models
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Generate Data
```bash
# Basic behavioral data (2,500 trials)
python data_generation/generate_data.py

# Enhanced psychological data (12,500 trials)
python data_generation/generate_psychological_data.py
```

### 2. Train Models

#### Standard Choice2Vec
```bash
# Quick training (1000 epochs)
python standard_choice2vec/train_choice2vec.py

# Compare different loss functions
python standard_choice2vec/compare_losses.py
```

#### Disentangled Choice2Vec
```bash
# Test implementation
python disentangled_choice2vec/test_disentanglement.py

# Full comparison training
python disentangled_choice2vec/train_disentangled_choice2vec.py
```

#### Long-term Training with Checkpoints
```bash
# 150,000 epoch training with automatic checkpoints
python checkpoints/psychological_unsupervised_classification.py
```

### 3. Evaluate Models
```bash
# List available saved models
python evaluation/list_models.py

# Evaluate a specific model
python evaluation/evaluate_saved_model.py

# Analyze learning curves
python evaluation/analyze_learning_curves.py
```

## 📊 Approaches Implemented

### 1. **Standard Choice2Vec** (`standard_choice2vec/`)
- Base wav2vec 2.0 adaptation for behavioral data
- InfoNCE contrastive learning
- Selective masking strategy
- Multiple loss function variants

**Key Features:**
- 3.9M parameters
- Transformer-based architecture
- Product quantization
- Achieves ~84% psychological state classification

### 2. **Disentangled Choice2Vec** (`disentangled_choice2vec/`)
- Enhanced with disentanglement techniques
- β-VAE style regularization
- Mutual information minimization
- Orthogonality constraints
- Factor-wise contrastive learning

**Key Features:**
- Addresses entangled representations (10+ PCs → 3-5 PCs for 95% variance)
- 4 quantizer groups for finer factorization
- Multiple disentanglement loss terms
- Comprehensive analysis tools

### 3. **Checkpoint System** (`checkpoints/`)
- Long-term training (150,000 epochs)
- Automatic checkpointing every 10,000 epochs
- Comprehensive analysis at each checkpoint
- Multiple clustering algorithms
- Extensive visualization generation

**Key Features:**
- Saves model weights, representations, and clustering results
- Automatic PCA, t-SNE, and clustering analysis
- 4-row visualization plots
- Cross-device compatible saving

## 🔧 Installation

```bash
# Create conda environment
conda create -n choice2vec python=3.9
conda activate choice2vec

# Install dependencies
pip install jax[cuda] flax optax pandas numpy matplotlib seaborn scikit-learn

# For CPU-only (if no GPU)
pip install jax[cpu] flax optax pandas numpy matplotlib seaborn scikit-learn
```

## 📖 Documentation

- **[Main Documentation](docs/README.md)**: Complete technical documentation
- **[Disentanglement Guide](docs/DISENTANGLEMENT_README.md)**: Disentanglement techniques and theory
- **[Checkpoint Guide](docs/CHECKPOINT_README.md)**: Long-term training system

## 🎯 Key Results

### Standard Choice2Vec
- **Accuracy**: 83.8% psychological state classification
- **Training**: Converges in ~1000 epochs
- **Architecture**: 3.9M parameters, 4 transformer layers

### Disentangled Choice2Vec
- **PCA Efficiency**: Reduces components needed for 95% variance from 10+ to 3-5
- **Factor Independence**: Mean correlation between factors < 0.3
- **Interpretability**: Each factor captures distinct behavioral aspects

### Long-term Training
- **Duration**: 150,000 epochs with checkpoints
- **Analysis**: Comprehensive clustering and representation analysis
- **Visualization**: Automatic generation of analysis plots

## 🔬 Research Applications

- **Cognitive Science**: Understanding decision-making processes
- **Psychology**: Modeling attention and engagement states
- **Human-Computer Interaction**: Adaptive interfaces based on user state
- **Neuroscience**: Computational models of behavioral patterns

## 🤝 Contributing

The codebase is organized for easy extension:

1. **New Models**: Add to appropriate folder with consistent API
2. **New Analysis**: Add to `evaluation/` folder
3. **New Data**: Add generators to `data_generation/`
4. **Documentation**: Update relevant README files

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Based on wav2vec 2.0 principles adapted for behavioral data analysis. Incorporates techniques from β-VAE, Factor-VAE, and contrastive learning literature. 