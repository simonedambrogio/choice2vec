import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from functools import partial

# All components are now self-contained in this file

# Configure JAX for GPU usage
print("Available JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

# Enable GPU memory preallocation for better performance
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def simple_contrastive_loss(projected_features: jnp.ndarray,
                           quantized_features: jnp.ndarray,
                           mask: jnp.ndarray,
                           temperature: float = 0.1) -> jnp.ndarray:
    """
    Simple, JAX-compatible contrastive loss using MSE between projected and quantized features.
    
    This is a simplified version that avoids complex indexing issues while still providing
    the core learning signal: align projected contextualized features with quantized targets.
    
    Args:
        projected_features: [batch_size, seq_len, embed_dim] from projection head
        quantized_features: [batch_size, seq_len, embed_dim] quantized targets
        mask: [batch_size, seq_len] mask indicating which positions were masked
        temperature: Not used in this simple version
    
    Returns:
        Contrastive loss (MSE between projected and quantized at masked positions)
    """
    # Check if any positions are masked
    num_masked = jnp.sum(mask)
    
    def no_mask_loss():
        return jnp.array(0.0)
    
    def compute_mse_loss():
        # Expand mask to match feature dimensions
        mask_expanded = mask[..., None]  # [batch_size, seq_len, 1]
        
        # Compute MSE between projected and quantized features
        diff = projected_features - quantized_features
        squared_diff = jnp.sum(diff ** 2, axis=-1)  # [batch_size, seq_len]
        
        # Apply mask and average over masked positions
        masked_loss = squared_diff * mask
        total_loss = jnp.sum(masked_loss) / (num_masked + 1e-8)
        
        return total_loss
    
    return jax.lax.cond(num_masked > 0, compute_mse_loss, no_mask_loss)


def cosine_contrastive_loss(projected_features: jnp.ndarray,
                           quantized_features: jnp.ndarray,
                           mask: jnp.ndarray,
                           temperature: float = 0.1) -> jnp.ndarray:
    """
    Cosine similarity-based contrastive loss (simpler than InfoNCE).
    
    Maximizes cosine similarity between projected and quantized features at masked positions.
    """
    num_masked = jnp.sum(mask)
    
    def no_mask_loss():
        return jnp.array(0.0)
    
    def compute_cosine_loss():
        # Normalize features
        proj_norm = projected_features / (jnp.linalg.norm(projected_features, axis=-1, keepdims=True) + 1e-8)
        quant_norm = quantized_features / (jnp.linalg.norm(quantized_features, axis=-1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        cosine_sim = jnp.sum(proj_norm * quant_norm, axis=-1)  # [batch_size, seq_len]
        
        # We want to maximize similarity, so minimize negative similarity
        negative_sim = -cosine_sim
        
        # Apply mask and average
        masked_loss = negative_sim * mask
        total_loss = jnp.sum(masked_loss) / (num_masked + 1e-8)
        
        return total_loss
    
    return jax.lax.cond(num_masked > 0, compute_cosine_loss, no_mask_loss)


class BehavioralFeatureEncoder(nn.Module):
    """
    Feature encoder that maps behavioral features to latent representations.
    Handles behavioral and environmental features separately.
    """
    hidden_dims: Tuple[int, ...] = (64, 128, 256)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, behavioral_features, environmental_features, training: bool = True):
        # Concatenate behavioral and environmental features
        combined_features = jnp.concatenate([behavioral_features, environmental_features], axis=-1)
        
        x = combined_features
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        return x


class ProductQuantizer(nn.Module):
    """
    Product quantization module that creates discrete behavioral codes.
    """
    num_groups: int = 2
    num_entries_per_group: int = 128
    embed_dim: int = 256
    
    @nn.compact
    def __call__(self, features, training: bool = True):
        batch_size, seq_len, feature_dim = features.shape
        
        # Project to logits for each group
        group_dim = feature_dim // self.num_groups
        
        # Split features into groups
        grouped_features = features.reshape(batch_size, seq_len, self.num_groups, group_dim)
        
        # Create codebooks for each group
        codebooks = []
        quantized_groups = []
        indices_groups = []
        
        for g in range(self.num_groups):
            # Project group features to logits
            group_features = grouped_features[:, :, g, :]  # [batch, seq, group_dim]
            logits = nn.Dense(self.num_entries_per_group)(group_features)  # [batch, seq, num_entries]
            
            # Gumbel softmax for differentiable discrete sampling
            if training:
                gumbel_noise = jax.random.gumbel(
                    self.make_rng('gumbel'), logits.shape
                )
                logits_with_noise = logits + gumbel_noise
            else:
                logits_with_noise = logits
            
            # Softmax and straight-through estimator
            soft_indices = nn.softmax(logits_with_noise, axis=-1)
            hard_indices = jnp.argmax(logits_with_noise, axis=-1)
            
            # Create one-hot encoding for straight-through estimator
            hard_one_hot = jax.nn.one_hot(hard_indices, self.num_entries_per_group)
            
            # Straight-through estimator: use hard in forward, soft in backward
            indices_one_hot = jax.lax.stop_gradient(hard_one_hot - soft_indices) + soft_indices
            indices_groups.append(hard_indices)
            
            # Create codebook
            codebook = self.param(
                f'codebook_{g}',
                nn.initializers.normal(stddev=0.02),
                (self.num_entries_per_group, group_dim)
            )
            codebooks.append(codebook)
            
            # Get quantized features using straight-through estimator
            quantized = jnp.einsum('bsn,nd->bsd', indices_one_hot, codebook)
            quantized_groups.append(quantized)
        
        # Concatenate quantized groups
        quantized_features = jnp.concatenate(quantized_groups, axis=-1)
        quantized_indices = jnp.stack(indices_groups, axis=-1)
        
        # Project to final embedding dimension
        quantized_features = nn.Dense(self.embed_dim)(quantized_features)
        
        return quantized_features, quantized_indices


class RelativePositionalEmbedding(nn.Module):
    """
    Relative positional embedding using 1D convolution.
    """
    embed_dim: int = 256
    kernel_size: int = 128
    
    @nn.compact
    def __call__(self, x):
        # Apply 1D convolution for relative positional information
        conv_out = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.kernel_size,),
            padding='SAME',
            feature_group_count=min(16, self.embed_dim)
        )(x)
        
        conv_out = nn.gelu(conv_out)
        
        # Add to input and apply layer norm
        x = x + conv_out
        x = nn.LayerNorm()(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention.
    """
    embed_dim: int = 256
    num_heads: int = 4
    mlp_dim: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Multi-head self-attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=not training
        )(x, x)
        
        # Residual connection and layer norm
        x = nn.LayerNorm()(x + attn_out)
        
        # MLP
        mlp_out = nn.Dense(self.mlp_dim)(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_out)
        mlp_out = nn.Dense(self.embed_dim)(mlp_out)
        
        # Residual connection and layer norm
        x = nn.LayerNorm()(x + mlp_out)
        
        return x


class Choice2Vec(nn.Module):
    """
    Complete Choice2Vec model for self-supervised learning of behavioral representations.
    """
    encoder_hidden_dims: Tuple[int, ...] = (64, 128, 256)
    num_quantizer_groups: int = 2
    num_entries_per_group: int = 128
    num_transformer_layers: int = 4
    embed_dim: int = 256
    num_heads: int = 4
    dropout_rate: float = 0.1
    mask_prob: float = 0.15
    
    def setup(self):
        self.feature_encoder = BehavioralFeatureEncoder(
            hidden_dims=self.encoder_hidden_dims,
            dropout_rate=self.dropout_rate
        )
        
        self.quantizer = ProductQuantizer(
            num_groups=self.num_quantizer_groups,
            num_entries_per_group=self.num_entries_per_group,
            embed_dim=self.embed_dim
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
        
        self.projection_head = nn.Dense(self.embed_dim)
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
        
        # 2. Quantize features (for targets)
        quantized_features, quantized_indices = self.quantizer(
            latent_features, training=training
        )
        
        # 3. Create mask for behavioral features only
        if mask_key is not None:
            mask = self.create_mask(mask_key, batch_size, seq_len)
        else:
            mask = jnp.zeros((batch_size, seq_len), dtype=bool)
        
        # 4. Apply selective masking (only to behavioral part of latent features)
        mask_expanded = mask[..., None]  # [batch_size, seq_len, 1]
        
        # Only mask the latent features, not the environmental context
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
        
        # 7. Project features for contrastive learning
        projected_features = self.projection_head(contextualized_features)
        
        # 8. Predict behavioral features at masked positions
        behavioral_predictions = self.behavioral_predictor(contextualized_features)
        
        # 9. Create targets for behavioral prediction (original behavioral features)
        behavioral_targets = behavioral_features
        
        return {
            'contextualized_features': contextualized_features,
            'projected_features': projected_features,
            'quantized_features': quantized_features,
            'quantized_indices': quantized_indices,
            'behavioral_predictions': behavioral_predictions,
            'behavioral_targets': behavioral_targets,
            'mask': mask
        }


def behavioral_prediction_loss(predictions: jnp.ndarray, 
                             targets: jnp.ndarray, 
                             mask: jnp.ndarray) -> jnp.ndarray:
    """Compute MSE loss for behavioral prediction at masked positions."""
    num_masked = jnp.sum(mask)
    
    def no_mask_loss():
        return jnp.array(0.0)
    
    def compute_loss():
        # Compute MSE between predictions and targets
        diff = predictions - targets
        squared_diff = jnp.sum(diff ** 2, axis=-1)  # Sum over behavioral features
        
        # Apply mask and average over masked positions
        masked_loss = squared_diff * mask
        total_loss = jnp.sum(masked_loss) / (num_masked + 1e-8)
        
        return total_loss
    
    return jax.lax.cond(num_masked > 0, compute_loss, no_mask_loss)


def diversity_loss(quantized_indices: jnp.ndarray, 
                  num_groups: int, 
                  num_entries: int) -> jnp.ndarray:
    """Encourage uniform usage of codebook entries."""
    total_loss = 0.0
    
    for g in range(num_groups):
        indices = quantized_indices[:, :, g]  # [batch_size, seq_len]
        
        # Compute histogram of indices
        hist = jnp.zeros(num_entries)
        for i in range(num_entries):
            hist = hist.at[i].set(jnp.sum(indices == i))
        
        # Normalize to probabilities
        probs = hist / (jnp.sum(hist) + 1e-8)
        
        # Compute entropy (we want to maximize it, so minimize negative entropy)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
        max_entropy = jnp.log(num_entries)
        
        # Convert to loss (minimize negative normalized entropy)
        group_loss = -(entropy / max_entropy)
        total_loss += group_loss
    
    return total_loss / num_groups


def wav2vec2_contrastive_loss(projected_features: jnp.ndarray,
                             quantized_features: jnp.ndarray,
                             mask: jnp.ndarray,
                             temperature: float = 0.1,
                             num_negatives: int = 100,
                             rng_key: jax.random.PRNGKey = None) -> jnp.ndarray:
    """
    wav2vec 2.0 InfoNCE contrastive loss with negative sampling.
    
    Simplified JAX-compatible implementation that approximates the wav2vec 2.0 loss
    using vectorized operations to avoid JIT compilation issues.
    
    Args:
        projected_features: [batch_size, seq_len, embed_dim] contextualized features
        quantized_features: [batch_size, seq_len, embed_dim] quantized targets
        mask: [batch_size, seq_len] mask indicating which positions were masked
        temperature: temperature parameter κ for scaling logits
        num_negatives: number K of negative samples per positive (not used in this simplified version)
        rng_key: random key for negative sampling
        
    Returns:
        InfoNCE-style contrastive loss
    """
    batch_size, seq_len, embed_dim = projected_features.shape
    num_masked = jnp.sum(mask)
    
    def no_mask_loss():
        return jnp.array(0.0)
    
    def compute_infonce_loss():
        # Normalize features for cosine similarity
        def normalize(x):
            return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        
        proj_norm = normalize(projected_features)  # [batch, seq, embed_dim]
        quant_norm = normalize(quantized_features)  # [batch, seq, embed_dim]
        
        # Compute positive similarities (element-wise)
        positive_similarities = jnp.sum(proj_norm * quant_norm, axis=-1)  # [batch, seq]
        
        # For negatives, we'll use a simplified approach:
        # Compare each position with all other positions in the batch
        # Reshape for matrix multiplication
        proj_flat = proj_norm.reshape(-1, embed_dim)  # [batch*seq, embed_dim]
        quant_flat = quant_norm.reshape(-1, embed_dim)  # [batch*seq, embed_dim]
        
        # Compute all pairwise similarities
        all_similarities = jnp.dot(proj_flat, quant_flat.T)  # [batch*seq, batch*seq]
        
        # Apply temperature scaling
        positive_logits = positive_similarities / temperature  # [batch, seq]
        all_logits = all_similarities / temperature  # [batch*seq, batch*seq]
        
        # For each masked position, compute InfoNCE loss
        # We'll use a simplified approach: for each position, use all other positions as negatives
        mask_flat = mask.reshape(-1)  # [batch*seq]
        
        # Get diagonal (positive) logits
        positive_logits_flat = jnp.diag(all_logits)  # [batch*seq]
        
        # For each row, compute logsumexp over all columns (including positive)
        log_sum_exp_all = jax.scipy.special.logsumexp(all_logits, axis=1)  # [batch*seq]
        
        # InfoNCE loss: -log(exp(positive) / sum(exp(all)))
        # = -positive + log_sum_exp(all)
        infonce_losses = log_sum_exp_all - positive_logits_flat  # [batch*seq]
        
        # Apply mask and average over masked positions
        masked_losses = infonce_losses * mask_flat
        total_loss = jnp.sum(masked_losses) / (num_masked + 1e-8)
        
        return total_loss
    
    return jax.lax.cond(num_masked > 0, compute_infonce_loss, no_mask_loss)


class Choice2VecTrainer:
    """
    Trainer for Choice2Vec with complete self-supervised learning.
    """
    
    def __init__(self, 
                 model: Choice2Vec,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 diversity_weight: float = 0.1,
                 contrastive_weight: float = 1.0,
                 use_cosine_loss: bool = True,
                 use_wav2vec2_loss: bool = False,
                 temperature: float = 0.1,
                 num_negatives: int = 100):
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.diversity_weight = diversity_weight
        self.contrastive_weight = contrastive_weight
        self.use_cosine_loss = use_cosine_loss
        self.use_wav2vec2_loss = use_wav2vec2_loss
        self.temperature = temperature
        self.num_negatives = num_negatives
        
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
        """Single training step with complete loss function."""
        
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
            
            # 2. Contrastive loss
            if self.use_wav2vec2_loss:
                # Use wav2vec 2.0 InfoNCE loss
                contrastive_loss_val = wav2vec2_contrastive_loss(
                    outputs['projected_features'],
                    outputs['quantized_features'],
                    outputs['mask'],
                    temperature=self.temperature,
                    num_negatives=self.num_negatives,
                    rng_key=rng1
                )
            elif self.use_cosine_loss:
                contrastive_loss_val = cosine_contrastive_loss(
                    outputs['projected_features'],
                    outputs['quantized_features'],
                    outputs['mask']
                )
            else:
                contrastive_loss_val = simple_contrastive_loss(
                    outputs['projected_features'],
                    outputs['quantized_features'],
                    outputs['mask']
                )
            
            # 3. Diversity loss
            div_loss = diversity_loss(
                outputs['quantized_indices'],
                self.model.num_quantizer_groups,
                self.model.num_entries_per_group
            )
            
            # Total loss
            total_loss = (behavioral_loss + 
                         self.contrastive_weight * contrastive_loss_val + 
                         self.diversity_weight * div_loss)
            
            return total_loss, {
                'total_loss': total_loss,
                'behavioral_loss': behavioral_loss,
                'contrastive_loss': contrastive_loss_val,
                'diversity_loss': div_loss,
                'mask_ratio': jnp.mean(outputs['mask'])
            }
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, metrics


def prepare_behavioral_data(df: pd.DataFrame, 
                           window_size: int = 50, 
                           stride: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare behavioral data for Choice2Vec training.
    
    Handles both data formats:
    - New psychological format: choice_correct (0/1), rt, trial_in_subtask, subtask
    - Old behavioral format: choice (0/1), rt, trial_in_subtask, subtask, value_difference
    
    Returns:
        behavioral_features: [num_windows, window_size, 2] (choice/choice_correct, rt)
        environmental_features: [num_windows, window_size, 2-3] (trial_in_subtask, subtask, [value_difference])
        states: [num_windows, window_size] (engagement states)
    """
    
    # Detect data format and define feature columns
    if 'choice_correct' in df.columns:
        # New psychological data format
        behavioral_cols = ['choice_correct', 'rt']
        environmental_cols = ['trial_in_subtask', 'subtask']
        print("   Using psychological data format (choice_correct, no value_difference)")
    else:
        # Old behavioral data format
        behavioral_cols = ['choice', 'rt']
        environmental_cols = ['trial_in_subtask', 'subtask', 'value_difference']
        print("   Using standard data format (choice, with value_difference)")
    
    state_col = 'psychological_state'
    
    # Normalize features
    df_norm = df.copy()
    
    # Normalize behavioral features
    for col in behavioral_cols:
        df_norm[col] = (df_norm[col] - df_norm[col].mean()) / (df_norm[col].std() + 1e-8)
    
    # Normalize environmental features
    for col in environmental_cols:
        df_norm[col] = (df_norm[col] - df_norm[col].mean()) / (df_norm[col].std() + 1e-8)
    
    # Convert psychological state to numeric (engaged=1, disengaged=0)
    df_norm[state_col] = (df_norm[state_col] == 'engaged').astype(float)
    
    # Create sliding windows
    behavioral_windows = []
    environmental_windows = []
    state_windows = []
    
    for start_idx in range(0, len(df_norm) - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Extract windows
        behavioral_window = df_norm[behavioral_cols].iloc[start_idx:end_idx].values
        environmental_window = df_norm[environmental_cols].iloc[start_idx:end_idx].values
        state_window = df_norm[state_col].iloc[start_idx:end_idx].values
        
        behavioral_windows.append(behavioral_window)
        environmental_windows.append(environmental_window)
        state_windows.append(state_window)
    
    return (np.array(behavioral_windows), 
            np.array(environmental_windows), 
            np.array(state_windows))

