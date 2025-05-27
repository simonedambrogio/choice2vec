# Disentangled Choice2Vec: Addressing Entangled Representations

This document explains the disentanglement techniques implemented to address the issue of entangled representations in Choice2Vec, where 10+ PCs are needed to explain 95% of variance.

## The Entanglement Problem

**Observation:** Standard Choice2Vec requires ~10 principal components to explain 95% of variance, suggesting highly entangled representations where:
- Multiple dimensions encode overlapping information
- Individual factors of variation are not cleanly separated
- Downstream tasks may struggle to isolate relevant features

**Goal:** Learn disentangled representations where:
- Each factor captures independent aspects of the data
- Fewer PCs explain most variance (concentrated information)
- Individual factors are interpretable and useful

## Disentanglement Techniques Implemented

### 1. β-VAE Style Quantizer Regularization

**Concept:** Based on β-VAE (Higgins et al., 2017), adds a commitment loss that encourages the quantizer to learn structured, disentangled codes.

```python
class DisentangledProductQuantizer(nn.Module):
    beta: float = 4.0  # β-VAE regularization strength
    
    def __call__(self, features, training=True):
        # ... quantization process ...
        
        # Commitment loss (β-VAE style)
        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - projected) ** 2)
        total_commitment_loss = self.beta * commitment_loss
```

**How it works:**
- Forces the encoder to commit to discrete codes
- β parameter controls the pressure for disentanglement
- Higher β → more disentangled but potentially less expressive representations

### 2. Mutual Information Minimization

**Concept:** Minimize mutual information between different groups of features to encourage independence.

```python
def mutual_information_loss(features, num_groups=4, temperature=0.1):
    # Split features into groups
    feature_groups = jnp.split(features, num_groups, axis=-1)
    
    # Compute MI between all pairs of groups
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            # Minimize correlation between groups
            similarity = jnp.dot(group_i_norm, group_j_norm.T) / temperature
            identity_target = jnp.eye(similarity.shape[0])
            mi_loss = jnp.mean((similarity - identity_target) ** 2)
```

**How it works:**
- Splits representation into groups (factors)
- Penalizes correlation between different factors
- Encourages each factor to capture unique information

### 3. Orthogonality Constraints

**Concept:** Stronger independence constraint that enforces orthogonality between feature groups.

```python
def orthogonality_loss(features, num_groups=4):
    # Split into groups
    feature_groups = jnp.split(features_flat, num_groups, axis=-1)
    
    # Compute cross-correlation matrix between groups
    cross_corr = jnp.dot(group_i.T, group_j)  # [group_dim, group_dim]
    
    # Minimize off-diagonal elements (enforce orthogonality)
    ortho_loss = jnp.sum(cross_corr ** 2)
```

**How it works:**
- Enforces orthogonality between factor groups
- Stronger constraint than MI minimization
- Ensures factors are linearly independent

### 4. Factor-wise Contrastive Learning

**Concept:** Apply contrastive learning separately to each factor, encouraging specialization.

```python
def factor_contrastive_loss(projected_features, quantized_features, mask, num_groups=4):
    # Split features into groups
    proj_groups = jnp.split(projected_features, num_groups, axis=-1)
    quant_groups = jnp.split(quantized_features, num_groups, axis=-1)
    
    # Apply InfoNCE loss to each group separately
    for proj_group, quant_group in zip(proj_groups, quant_groups):
        # Contrastive learning within this factor only
        group_loss = infonce_loss(proj_group, quant_group, mask)
```

**How it works:**
- Each factor learns its own contrastive objective
- Prevents factors from learning redundant information
- Encourages factor specialization

### 5. Enhanced Architecture Design

**Key Changes:**
- **More quantizer groups:** 4 groups instead of 2 (finer factorization)
- **Smaller codebooks:** 64 entries per group instead of 128 (forces compression)
- **Factor-specific projections:** Separate projection heads for each factor
- **Global + factor projections:** Both global and factor-wise contrastive learning

## Complete Loss Function

The disentangled model combines multiple loss terms:

```python
total_loss = (
    behavioral_loss +                           # Task-specific learning
    contrastive_weight * global_contrastive_loss +     # Global representation quality
    factor_contrastive_weight * factor_contrastive_loss + # Factor specialization
    mi_weight * mi_loss +                       # Factor independence (MI)
    orthogonality_weight * ortho_loss +         # Factor independence (orthogonality)
    commitment_weight * commitment_loss +       # β-VAE regularization
    diversity_weight * div_loss                 # Codebook diversity
)
```

**Weight Recommendations:**
- `contrastive_weight = 1.0` (baseline)
- `factor_contrastive_weight = 0.5` (moderate factor specialization)
- `mi_weight = 0.1` (light independence pressure)
- `orthogonality_weight = 0.05` (light orthogonality pressure)
- `commitment_weight = 1.0` (β-VAE regularization)
- `diversity_weight = 0.1` (codebook diversity)

## Expected Improvements

### 1. PCA Efficiency
**Before:** 10+ components for 95% variance
**After:** 3-5 components for 95% variance

### 2. Factor Independence
**Metric:** Mean correlation between factors
**Target:** < 0.3 (low correlation indicates independence)

### 3. Factor Interpretability
Each factor should capture distinct aspects:
- **Factor 1:** Choice patterns (accuracy, consistency)
- **Factor 2:** Response time patterns (speed, variability)
- **Factor 3:** Learning dynamics (adaptation, meta-learning)
- **Factor 4:** Context sensitivity (subtask effects)

### 4. Downstream Performance
**Expectation:** Maintained or improved classification accuracy with more interpretable features

## Usage

### Basic Training
```python
from disentangled_choice2vec import DisentangledChoice2Vec, DisentangledChoice2VecTrainer

# Create disentangled model
model = DisentangledChoice2Vec(
    num_quantizer_groups=4,      # More factors
    num_entries_per_group=64,    # Smaller codebooks
    beta=4.0                     # β-VAE strength
)

# Create trainer with disentanglement losses
trainer = DisentangledChoice2VecTrainer(
    model=model,
    factor_contrastive_weight=0.5,
    mi_weight=0.1,
    orthogonality_weight=0.05,
    commitment_weight=1.0
)
```

### Comparison Training
```python
# Compare standard vs disentangled
python train_disentangled_choice2vec.py
```

This will:
- Train both standard and disentangled models
- Compare PCA efficiency
- Analyze factor independence
- Evaluate downstream task performance
- Generate comprehensive comparison plots

### Disentanglement Analysis
```python
from disentangled_choice2vec import analyze_disentanglement

# Analyze representation quality
metrics = analyze_disentanglement(representations, true_states, num_groups=4)

print(f"Factor independence: {1.0 - metrics['mean_factor_correlation']:.3f}")
print(f"Best factor accuracy: {metrics['best_factor_accuracy']:.3f}")
print(f"Disentanglement score: {metrics['disentanglement_score']:.3f}")
```

## Theoretical Background

### β-VAE (Higgins et al., 2017)
- Adds β coefficient to VAE loss: `L = reconstruction_loss + β * KL_divergence`
- Higher β encourages disentangled latent representations
- Trade-off between reconstruction quality and disentanglement

### Factor-VAE (Kim & Mnih, 2018)
- Encourages statistical independence between latent factors
- Uses total correlation penalty to minimize redundancy
- Our MI loss is inspired by this approach

### Contrastive Disentanglement (Caron et al., 2020)
- Applies contrastive learning to encourage disentanglement
- Multi-crop strategy for different views of the same data
- Our factor-wise contrastive loss extends this to behavioral data

### Information Bottleneck Principle
- Compress representations while preserving task-relevant information
- Orthogonality constraints implement this by forcing efficient encoding
- Each factor must capture unique, non-redundant information

## Hyperparameter Tuning

### β (Beta) Parameter
- **Low (1.0-2.0):** Mild disentanglement pressure
- **Medium (3.0-5.0):** Balanced disentanglement/performance
- **High (6.0+):** Strong disentanglement, may hurt performance

### MI Weight
- **Low (0.05-0.1):** Gentle independence encouragement
- **Medium (0.1-0.3):** Moderate independence pressure
- **High (0.5+):** Strong independence, may be too restrictive

### Orthogonality Weight
- **Low (0.01-0.05):** Light orthogonality constraint
- **Medium (0.05-0.1):** Moderate orthogonality pressure
- **High (0.1+):** Strong orthogonality, may limit expressiveness

### Number of Groups
- **2 groups:** Basic factorization (choice vs RT)
- **4 groups:** Fine-grained factorization (recommended)
- **8+ groups:** Very fine factorization (may be over-parameterized)

## Evaluation Metrics

### 1. PCA Efficiency
```python
pca = PCA()
pca.fit(representations)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_95 = np.argmax(cumvar >= 0.95) + 1  # Components for 95% variance
```

### 2. Factor Independence
```python
correlations = []
for i in range(num_groups):
    for j in range(i + 1, num_groups):
        corr = np.corrcoef(factor_i, factor_j)[0, 1]
        correlations.append(abs(corr))
independence = 1.0 - np.mean(correlations)
```

### 3. Factor Separability
```python
# How well each factor separates psychological states
for factor in factors:
    clf = LogisticRegression()
    clf.fit(factor, true_states)
    accuracy = clf.score(factor, true_states)
```

### 4. Disentanglement Score
```python
disentanglement_score = independence_score * separability_score
```

## Expected Results

### Successful Disentanglement
- **PCA efficiency:** 3-5 components for 95% variance (vs 10+ for standard)
- **Factor independence:** Mean correlation < 0.3
- **Factor separability:** At least one factor achieves >70% accuracy
- **Disentanglement score:** > 0.5

### Potential Issues
- **Over-regularization:** Too much disentanglement pressure hurts performance
- **Under-regularization:** Insufficient pressure, still entangled
- **Factor collapse:** Multiple factors learn the same information
- **Performance trade-off:** Better disentanglement but worse downstream accuracy

## Integration with Checkpoint System

The disentangled model can be integrated with the checkpoint system:

```python
# Modify psychological_unsupervised_classification.py to use disentangled model
from disentangled_choice2vec import DisentangledChoice2Vec, DisentangledChoice2VecTrainer

# Replace standard model with disentangled version
model = DisentangledChoice2Vec(...)
trainer = DisentangledChoice2VecTrainer(...)

# All checkpoint functionality remains the same
# Additional disentanglement metrics will be saved and analyzed
```

This provides a comprehensive solution to the entangled representation problem while maintaining all the existing functionality and analysis capabilities. 