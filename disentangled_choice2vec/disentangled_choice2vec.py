#!/usr/bin/env python3
"""
Disentangled Choice2Vec: Enhanced with Disentanglement Techniques

This module implements several approaches to encourage disentangled representations
in the Choice2Vec contrastive learning framework:

1. β-VAE style regularization on the quantizer
2. Mutual Information minimization between representation dimensions
3. Orthogonality constraints on learned features
4. Factor-wise contrastive learning
5. Hierarchical disentanglement with multi-scale representations

Based on:
- β-VAE (Higgins et al., 2017)
- Factor-VAE (Kim & Mnih, 2018) 
- Disentangled Contrastive Learning (Caron et al., 2020)
- SwAV with multi-crop (Caron et al., 2020)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import Tuple, Dict, Any, Optional
from functools import partial

# Import base components
from core.choice2vec_model import (
    BehavioralFeatureEncoder, ProductQuantizer, RelativePositionalEmbedding,
    TransformerBlock, behavioral_prediction_loss, diversity_loss
)


class DisentangledProductQuantizer(nn.Module):
    """
    Enhanced Product Quantizer with β-VAE style regularization for disentanglement.
    """
    num_groups: int = 4  # More groups for finer factorization
    num_entries_per_group: int = 64  # Smaller codebooks per group
    embed_dim: int = 256
    beta: float = 4.0  # β-VAE regularization strength
    
    def setup(self):
        self.group_dim = self.embed_dim // self.num_groups
        
        # Separate codebooks for each group
        self.codebooks = [
            self.param(f'codebook_{i}', 
                      nn.initializers.normal(stddev=0.02),
                      (self.num_entries_per_group, self.group_dim))
            for i in range(self.num_groups)
        ]
        
        # Projection layers for each group (encourage specialization)
        self.group_projections = [
            nn.Dense(self.group_dim, name=f'proj_{i}')
            for i in range(self.num_groups)
        ]
        
        # Final projection to reconstruct full dimension
        self.output_projection = nn.Dense(self.embed_dim)
    
    def __call__(self, features, training: bool = True):
        batch_size, seq_len, _ = features.shape
        
        # Split features into groups
        feature_groups = jnp.split(features, self.num_groups, axis=-1)
        
        quantized_groups = []
        indices_groups = []
        commitment_losses = []
        
        for i, (group_features, projection, codebook) in enumerate(
            zip(feature_groups, self.group_projections, self.codebooks)
        ):
            # Project group features
            projected = projection(group_features)  # [batch, seq, group_dim]
            
            # Compute distances to codebook entries
            distances = jnp.sum(
                (projected[..., None, :] - codebook[None, None, :, :]) ** 2,
                axis=-1
            )  # [batch, seq, num_entries]
            
            # Get closest codebook indices
            indices = jnp.argmin(distances, axis=-1)  # [batch, seq]
            
            # Get quantized vectors
            quantized = codebook[indices]  # [batch, seq, group_dim]
            
            # Commitment loss (β-VAE style)
            commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - projected) ** 2)
            commitment_losses.append(commitment_loss)
            
            # Straight-through estimator
            quantized = projected + jax.lax.stop_gradient(quantized - projected)
            
            quantized_groups.append(quantized)
            indices_groups.append(indices)
        
        # Concatenate quantized groups
        quantized_features = jnp.concatenate(quantized_groups, axis=-1)
        
        # Final projection to ensure proper dimensionality
        quantized_features = self.output_projection(quantized_features)
        
        # Stack indices
        indices = jnp.stack(indices_groups, axis=-1)  # [batch, seq, num_groups]
        
        # Total commitment loss with β weighting
        total_commitment_loss = self.beta * jnp.mean(jnp.array(commitment_losses))
        
        return quantized_features, indices, total_commitment_loss


def mutual_information_loss(features: jnp.ndarray, 
                           num_groups: int = 4,
                           temperature: float = 0.1) -> jnp.ndarray:
    """
    Minimize mutual information between different groups of features.
    
    This encourages each group to capture independent factors of variation.
    """
    batch_size, seq_len, embed_dim = features.shape
    group_dim = embed_dim // num_groups
    
    # Split features into groups
    feature_groups = jnp.split(features, num_groups, axis=-1)  # List of [batch, seq, group_dim]
    
    total_mi_loss = 0.0
    
    # Compute MI between all pairs of groups
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            group_i = feature_groups[i].reshape(-1, group_dim)  # [batch*seq, group_dim]
            group_j = feature_groups[j].reshape(-1, group_dim)  # [batch*seq, group_dim]
            
            # Normalize features
            group_i_norm = group_i / (jnp.linalg.norm(group_i, axis=-1, keepdims=True) + 1e-8)
            group_j_norm = group_j / (jnp.linalg.norm(group_j, axis=-1, keepdims=True) + 1e-8)
            
            # Compute similarity matrix
            similarity = jnp.dot(group_i_norm, group_j_norm.T) / temperature  # [batch*seq, batch*seq]
            
            # MI approximation: minimize correlation between groups
            # We want the similarity to be close to identity (independent)
            identity_target = jnp.eye(similarity.shape[0])
            mi_loss = jnp.mean((similarity - identity_target) ** 2)
            
            total_mi_loss += mi_loss
    
    # Normalize by number of pairs
    num_pairs = (num_groups * (num_groups - 1)) // 2
    return total_mi_loss / num_pairs


def orthogonality_loss(features: jnp.ndarray, num_groups: int = 4) -> jnp.ndarray:
    """
    Encourage orthogonality between different feature groups.
    
    This is a stronger constraint than MI minimization.
    """
    batch_size, seq_len, embed_dim = features.shape
    group_dim = embed_dim // num_groups
    
    # Reshape for easier computation
    features_flat = features.reshape(-1, embed_dim)  # [batch*seq, embed_dim]
    
    # Split into groups
    feature_groups = jnp.split(features_flat, num_groups, axis=-1)  # List of [batch*seq, group_dim]
    
    total_ortho_loss = 0.0
    
    # Compute orthogonality loss between all pairs
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            group_i = feature_groups[i]  # [batch*seq, group_dim]
            group_j = feature_groups[j]  # [batch*seq, group_dim]
            
            # Compute cross-correlation matrix
            cross_corr = jnp.dot(group_i.T, group_j)  # [group_dim, group_dim]
            
            # Orthogonality loss: minimize off-diagonal elements
            ortho_loss = jnp.sum(cross_corr ** 2)
            total_ortho_loss += ortho_loss
    
    # Normalize
    num_pairs = (num_groups * (num_groups - 1)) // 2
    return total_ortho_loss / (num_pairs * batch_size * seq_len)


def factor_contrastive_loss(projected_features: jnp.ndarray,
                           quantized_features: jnp.ndarray,
                           mask: jnp.ndarray,
                           num_groups: int = 4,
                           temperature: float = 0.1) -> jnp.ndarray:
    """
    Factor-wise contrastive learning: apply contrastive loss to each factor separately.
    
    This encourages each factor to capture distinct aspects of the data.
    """
    num_masked = jnp.sum(mask)
    
    def no_mask_loss():
        return jnp.array(0.0)
    
    def compute_factor_contrastive():
        batch_size, seq_len, embed_dim = projected_features.shape
        group_dim = embed_dim // num_groups
        
        # Split features into groups
        proj_groups = jnp.split(projected_features, num_groups, axis=-1)
        quant_groups = jnp.split(quantized_features, num_groups, axis=-1)
        
        total_loss = 0.0
        
        for proj_group, quant_group in zip(proj_groups, quant_groups):
            # Normalize features
            proj_norm = proj_group / (jnp.linalg.norm(proj_group, axis=-1, keepdims=True) + 1e-8)
            quant_norm = quant_group / (jnp.linalg.norm(quant_group, axis=-1, keepdims=True) + 1e-8)
            
            # Compute positive similarities
            positive_sim = jnp.sum(proj_norm * quant_norm, axis=-1)  # [batch, seq]
            
            # Apply temperature and mask
            positive_logits = positive_sim / temperature
            
            # For negatives, use other positions in the same group
            proj_flat = proj_norm.reshape(-1, group_dim)  # [batch*seq, group_dim]
            quant_flat = quant_norm.reshape(-1, group_dim)  # [batch*seq, group_dim]
            
            # Compute all pairwise similarities within this group
            all_similarities = jnp.dot(proj_flat, quant_flat.T) / temperature  # [batch*seq, batch*seq]
            
            # InfoNCE loss for this group
            positive_logits_flat = jnp.diag(all_similarities)
            log_sum_exp_all = jax.scipy.special.logsumexp(all_similarities, axis=1)
            
            group_loss = log_sum_exp_all - positive_logits_flat
            
            # Apply mask
            mask_flat = mask.reshape(-1)
            masked_group_loss = jnp.sum(group_loss * mask_flat) / (num_masked + 1e-8)
            
            total_loss += masked_group_loss
        
        return total_loss / num_groups
    
    return jax.lax.cond(num_masked > 0, compute_factor_contrastive, no_mask_loss)


class DisentangledChoice2Vec(nn.Module):
    """
    Enhanced Choice2Vec with multiple disentanglement techniques.
    """
    encoder_hidden_dims: Tuple[int, ...] = (64, 128, 256)
    num_quantizer_groups: int = 4  # More groups for better factorization
    num_entries_per_group: int = 64  # Smaller codebooks
    num_transformer_layers: int = 4
    embed_dim: int = 256
    num_heads: int = 4
    dropout_rate: float = 0.1
    mask_prob: float = 0.15
    beta: float = 4.0  # β-VAE strength
    
    def setup(self):
        self.feature_encoder = BehavioralFeatureEncoder(
            hidden_dims=self.encoder_hidden_dims,
            dropout_rate=self.dropout_rate
        )
        
        # Use disentangled quantizer
        self.quantizer = DisentangledProductQuantizer(
            num_groups=self.num_quantizer_groups,
            num_entries_per_group=self.num_entries_per_group,
            embed_dim=self.embed_dim,
            beta=self.beta
        )
        
        self.positional_embedding = RelativePositionalEmbedding(
            embed_dim=self.embed_dim
        )
        
        self.transformer_layers = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_transformer_layers)
        ]
        
        # Separate projection heads for each factor
        self.factor_projection_heads = [
            nn.Dense(self.embed_dim // self.num_quantizer_groups, name=f'factor_proj_{i}')
            for i in range(self.num_quantizer_groups)
        ]
        
        # Global projection head
        self.global_projection_head = nn.Dense(self.embed_dim)
        
        self.behavioral_predictor = nn.Dense(2)  # Predict choice and RT
        
        # Initialize mask token
        self.mask_token = self.param('mask_token', nn.initializers.normal(stddev=0.02), (self.embed_dim,))
    
    def create_mask(self, rng_key, batch_size: int, seq_len: int):
        """Create random mask for behavioral features only."""
        mask_probs = jax.random.uniform(rng_key, (batch_size, seq_len))
        mask = mask_probs < self.mask_prob
        return mask
    
    def __call__(self, behavioral_features, environmental_features, 
                 training: bool = True, mask_key: Optional[jax.random.PRNGKey] = None):
        
        batch_size, seq_len, _ = behavioral_features.shape
        
        # 1. Encode features
        latent_features = self.feature_encoder(
            behavioral_features, environmental_features, training=training
        )
        
        # 2. Quantize features with disentanglement
        quantized_features, quantized_indices, commitment_loss = self.quantizer(
            latent_features, training=training
        )
        
        # 3. Create mask
        if mask_key is not None:
            mask = self.create_mask(mask_key, batch_size, seq_len)
        else:
            mask = jnp.zeros((batch_size, seq_len), dtype=bool)
        
        # 4. Apply selective masking
        mask_expanded = mask[..., None]
        masked_latent_features = jnp.where(
            mask_expanded,
            self.mask_token[None, None, :],
            latent_features
        )
        
        # 5. Add positional embeddings
        contextualized_features = self.positional_embedding(masked_latent_features)
        
        # 6. Apply transformer layers
        for transformer_layer in self.transformer_layers:
            contextualized_features = transformer_layer(
                contextualized_features, training=training
            )
        
        # 7. Factor-wise projections
        factor_projections = []
        group_dim = self.embed_dim // self.num_quantizer_groups
        
        for i, proj_head in enumerate(self.factor_projection_heads):
            # Extract factor-specific features
            start_idx = i * group_dim
            end_idx = (i + 1) * group_dim
            factor_features = contextualized_features[..., start_idx:end_idx]
            factor_proj = proj_head(factor_features)
            factor_projections.append(factor_proj)
        
        # Concatenate factor projections
        factor_projected_features = jnp.concatenate(factor_projections, axis=-1)
        
        # 8. Global projection
        global_projected_features = self.global_projection_head(contextualized_features)
        
        # 9. Behavioral prediction
        behavioral_predictions = self.behavioral_predictor(contextualized_features)
        behavioral_targets = behavioral_features
        
        return {
            'contextualized_features': contextualized_features,
            'factor_projected_features': factor_projected_features,
            'global_projected_features': global_projected_features,
            'quantized_features': quantized_features,
            'quantized_indices': quantized_indices,
            'behavioral_predictions': behavioral_predictions,
            'behavioral_targets': behavioral_targets,
            'mask': mask,
            'commitment_loss': commitment_loss
        }


class DisentangledChoice2VecTrainer:
    """
    Trainer for Disentangled Choice2Vec with enhanced loss functions.
    """
    
    def __init__(self, 
                 model: DisentangledChoice2Vec,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 diversity_weight: float = 0.1,
                 contrastive_weight: float = 1.0,
                 factor_contrastive_weight: float = 0.5,
                 mi_weight: float = 0.1,
                 orthogonality_weight: float = 0.05,
                 commitment_weight: float = 1.0,
                 temperature: float = 0.1):
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.diversity_weight = diversity_weight
        self.contrastive_weight = contrastive_weight
        self.factor_contrastive_weight = factor_contrastive_weight
        self.mi_weight = mi_weight
        self.orthogonality_weight = orthogonality_weight
        self.commitment_weight = commitment_weight
        self.temperature = temperature
        
        # Initialize optimizer
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    
    def create_train_state(self, rng: jax.random.PRNGKey, 
                          behavioral_shape: Tuple[int, ...],
                          environmental_shape: Tuple[int, ...]) -> train_state.TrainState:
        """Initialize training state."""
        dummy_behavioral = jnp.ones(behavioral_shape)
        dummy_environmental = jnp.ones(environmental_shape)
        
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        variables = self.model.init(
            {'params': rng1, 'gumbel': rng2, 'dropout': rng3}, 
            dummy_behavioral, dummy_environmental,
            training=True, mask_key=rng1
        )
        
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=self.optimizer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, 
                   state: train_state.TrainState, 
                   behavioral_batch: jnp.ndarray,
                   environmental_batch: jnp.ndarray,
                   rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """Enhanced training step with disentanglement losses."""
        
        def loss_fn(params):
            rng1, rng2, rng3 = jax.random.split(rng, 3)
            
            outputs = self.model.apply(
                {'params': params}, 
                behavioral_batch, environmental_batch,
                training=True, 
                mask_key=rng1,
                rngs={'gumbel': rng2, 'dropout': rng3}
            )
            
            # 1. Behavioral prediction loss
            behavioral_loss = behavioral_prediction_loss(
                outputs['behavioral_predictions'],
                outputs['behavioral_targets'],
                outputs['mask']
            )
            
            # 2. Global contrastive loss (standard)
            from core.choice2vec_model import cosine_contrastive_loss
            global_contrastive_loss = cosine_contrastive_loss(
                outputs['global_projected_features'],
                outputs['quantized_features'],
                outputs['mask']
            )
            
            # 3. Factor-wise contrastive loss
            factor_contrastive_loss_val = factor_contrastive_loss(
                outputs['factor_projected_features'],
                outputs['quantized_features'],
                outputs['mask'],
                num_groups=self.model.num_quantizer_groups,
                temperature=self.temperature
            )
            
            # 4. Mutual information loss (encourage independence)
            mi_loss = mutual_information_loss(
                outputs['contextualized_features'],
                num_groups=self.model.num_quantizer_groups,
                temperature=self.temperature
            )
            
            # 5. Orthogonality loss (stronger independence constraint)
            ortho_loss = orthogonality_loss(
                outputs['contextualized_features'],
                num_groups=self.model.num_quantizer_groups
            )
            
            # 6. Commitment loss (from β-VAE quantizer)
            commitment_loss_val = outputs['commitment_loss']
            
            # 7. Diversity loss
            div_loss = diversity_loss(
                outputs['quantized_indices'],
                self.model.num_quantizer_groups,
                self.model.num_entries_per_group
            )
            
            # Total loss with all disentanglement terms
            total_loss = (
                behavioral_loss + 
                self.contrastive_weight * global_contrastive_loss +
                self.factor_contrastive_weight * factor_contrastive_loss_val +
                self.mi_weight * mi_loss +
                self.orthogonality_weight * ortho_loss +
                self.commitment_weight * commitment_loss_val +
                self.diversity_weight * div_loss
            )
            
            return total_loss, {
                'total_loss': total_loss,
                'behavioral_loss': behavioral_loss,
                'global_contrastive_loss': global_contrastive_loss,
                'factor_contrastive_loss': factor_contrastive_loss_val,
                'mi_loss': mi_loss,
                'orthogonality_loss': ortho_loss,
                'commitment_loss': commitment_loss_val,
                'diversity_loss': div_loss,
                'mask_ratio': jnp.mean(outputs['mask'])
            }
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, metrics


def analyze_disentanglement(representations: jnp.ndarray, 
                           true_states: jnp.ndarray,
                           num_groups: int = 4) -> Dict[str, float]:
    """
    Analyze the quality of disentanglement in learned representations.
    
    Returns metrics including:
    - Factor utilization (how much each factor is used)
    - Independence score (correlation between factors)
    - Separability per factor
    """
    batch_size, seq_len, embed_dim = representations.shape
    group_dim = embed_dim // num_groups
    
    # Flatten representations
    repr_flat = representations.reshape(-1, embed_dim)  # [batch*seq, embed_dim]
    states_flat = true_states.reshape(-1)  # [batch*seq]
    
    # Split into factor groups
    factor_groups = jnp.split(repr_flat, num_groups, axis=-1)
    
    metrics = {}
    
    # 1. Factor utilization (variance per factor)
    factor_variances = []
    for i, factor in enumerate(factor_groups):
        var = jnp.var(factor, axis=0).mean()  # Average variance across dimensions
        factor_variances.append(float(var))
        metrics[f'factor_{i}_variance'] = float(var)
    
    metrics['variance_balance'] = float(jnp.std(jnp.array(factor_variances)))  # Lower is better
    
    # 2. Independence score (correlation between factors)
    correlations = []
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            factor_i_mean = jnp.mean(factor_groups[i], axis=-1)  # [batch*seq]
            factor_j_mean = jnp.mean(factor_groups[j], axis=-1)  # [batch*seq]
            
            corr = jnp.corrcoef(factor_i_mean, factor_j_mean)[0, 1]
            correlations.append(float(jnp.abs(corr)))
    
    metrics['mean_factor_correlation'] = float(jnp.mean(jnp.array(correlations)))  # Lower is better
    metrics['max_factor_correlation'] = float(jnp.max(jnp.array(correlations)))   # Lower is better
    
    # 3. Separability per factor (how well each factor separates psychological states)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    binary_states = (true_states == 'engaged').astype(int) if isinstance(true_states[0], str) else true_states
    binary_states_flat = binary_states.reshape(-1)
    
    factor_accuracies = []
    for i, factor in enumerate(factor_groups):
        try:
            # Train classifier on this factor alone
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(np.array(factor), np.array(binary_states_flat))
            
            # Evaluate
            pred = clf.predict(np.array(factor))
            acc = accuracy_score(np.array(binary_states_flat), pred)
            factor_accuracies.append(acc)
            metrics[f'factor_{i}_accuracy'] = acc
        except:
            factor_accuracies.append(0.0)
            metrics[f'factor_{i}_accuracy'] = 0.0
    
    metrics['best_factor_accuracy'] = float(jnp.max(jnp.array(factor_accuracies)))
    metrics['mean_factor_accuracy'] = float(jnp.mean(jnp.array(factor_accuracies)))
    
    # 4. Disentanglement score (combination of independence and separability)
    independence_score = 1.0 - metrics['mean_factor_correlation']  # Higher is better
    separability_score = metrics['best_factor_accuracy']  # Higher is better
    
    metrics['disentanglement_score'] = float(independence_score * separability_score)
    
    return metrics


if __name__ == "__main__":
    print("🧠 Disentangled Choice2Vec: Enhanced Representation Learning")
    print("=" * 70)
    print("Features:")
    print("• β-VAE style quantizer regularization")
    print("• Mutual information minimization between factors")
    print("• Orthogonality constraints")
    print("• Factor-wise contrastive learning")
    print("• Comprehensive disentanglement analysis")
    print("=" * 70) 