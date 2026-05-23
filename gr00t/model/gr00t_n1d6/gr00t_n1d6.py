from typing import Tuple, Optional

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, BasicTransformerBlock
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree
from gr00t.model.modules.pgd_module import PGDAdversarialInjector, CSPGDAdversarialInjector
from torch.distributions import Dirichlet
import copy
class CategorySpecificConditionNoiseSampler(nn.Module):
    """Generates noise parameters (mean & std) conditioned on VL features, state, and embodiment."""
    def __init__(
        self,
        max_state_dim,
        max_action_dim,
        action_horizon,
        max_num_embodiments,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        ff_inner_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        cross_attention_dim: Optional[int] = None,
        min_std: float = 0.2,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False
        self.action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.num_layers = num_layers

        # State encoder (shared or same structure as in main head)
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.inner_dim,
            output_dim=self.inner_dim,
        )

        # Position embedding for action horizon (query positions)
        self.position_embedding = nn.Embedding(self.action_horizon, self.inner_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Cross-attention Transformer layers (decoder-style)
        self.layers = nn.ModuleList([
            BasicTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout = dropout,
                cross_attention_dim = cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias = attention_bias,
                upcast_attention = upcast_attention,
                norm_elementwise_affine = True,
                norm_type = "layer_norm",
                norm_eps = norm_eps,
                final_dropout = final_dropout,
                ff_inner_dim = ff_inner_dim,
            )
            for _ in range(self.num_layers)
        ])
        self.out_norm = nn.LayerNorm(self.inner_dim, eps=norm_eps)
        # Output heads: mean and log_var (both conditioned on embodiment)
        self.mean_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=self.inner_dim,
            hidden_dim=self.inner_dim,
            output_dim=1,
        )

        self.log_var_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=self.inner_dim,
            hidden_dim=self.inner_dim,
            output_dim=1,
        )

        # learnable temperature or min/max clamp for std
        self.min_std = min_std
        self.max_std = max_std

    def forward(self, vl_embeds, vl_attn_mask, state, embodiment_id, return_params=False, return_std_penalty=False):
        """
        Args:
            vl_embeds: [B, L, D] — vision-language embeddings (keys/values)
            state: [B, s_h, state_dim]
            embodiment_id: [B]
            return_params: if True, also return mean/std (useful for debugging)

        Returns:
            noise: [B, action_horizon, action_dim]
            (optional) mean, std: same shape
        """
        B = state.shape[0]
        device = state.device

        # Encode state → [B, D]
        state_emb = self.state_encoder(state, embodiment_id)  # [B, s_h, D]

        # Expand to [B, action_horizon, D]
        query = state_emb.expand(-1, self.action_horizon, -1).contiguous()

        # Add position embedding
        pos_ids = torch.arange(self.action_horizon, device=device)  # [H]
        pos_emb = self.position_embedding(pos_ids).unsqueeze(0)     # [1, H, D]
        query = query + pos_emb  # [B, H, D]
        # Pass through K transformer decoder layers
        for layer in self.layers:
            # Note: nn.TransformerDecoderLayer expects (tgt, memory)
            query = layer(
                hidden_states=query,
                attention_mask= None,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
            )
        query = self.out_norm(query)
        # Decode mean and log_var per timestep and embodiment
        mean = self.mean_decoder(query, embodiment_id)  # [B, H, action_dim]
        log_var = self.log_var_decoder(query, embodiment_id)  # [B, H, action_dim]

        mean = torch.tanh(mean)
        # Clamp log_var for numerical stability
        std = torch.exp(log_var/2)
        std = torch.clamp(std, self.min_std, self.max_std)

        # Reparameterization trick: ε ~ N(0, I)
        eps = torch.randn((state.shape[0], self.action_horizon, self.action_dim), device=device, dtype=state.dtype)  # [B, H, action_dim]
        noise = mean + std * eps      # [B, H, action_dim]
        out_dict = {
            "noise": noise
        }
        if return_params:
            out_dict["mean"] = mean
            out_dict["std"] = std
        if return_std_penalty:
            std_penalty = F.relu(1 - std).mean()
            out_dict["std_penalty"] = std_penalty
        return out_dict
    def expand_action_dimension(self, old_action_dim, new_action_dim):
        self.mean_decoder.expand_action_dimension(old_action_dim, new_action_dim, expand_input=False, expand_output=True)
        self.log_var_decoder.expand_action_dimension(old_action_dim, new_action_dim, expand_input=False, expand_output=True)
class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )
        if config.use_pgd and config.use_consistency_learning:
            self.pgd_adv_injector = CSPGDAdversarialInjector(
                epsilon = 0.03,
                alpha = 0.01,
                num_steps = 1,
                norm = "l2",  # "l2" or "linf"
                perturb_backbone_keys = ["backbone_features"],   # e.g., ["backbone_features"]
                perturb_action_keys = ["state"],     # e.g., ["state"]
                stop_gradient = False,
                w_pertub_loss = 0.5,
                perturb_noise = False,
                all_step_loss = True,
            )
        elif config.use_pgd:
            self.pgd_adv_injector = PGDAdversarialInjector(
                epsilon = 0.03,
                alpha = 0.01,
                num_steps = 3,
                norm = "l2",  # "l2" or "linf"
                perturb_backbone_keys = ["backbone_features"],   # e.g., ["backbone_features"]
                perturb_action_keys = ["state"],     # e.g., ["state"]
                stop_gradient = False,
                w_pertub_loss = 0.5,
                perturb_noise = False,
                all_step_loss = True,
            )
        if config.use_consistency_learning:
            self.register_buffer('action_loss_ema', torch.tensor(100.0))
        self.v_norm_ratio_ema = 0
        self.train_steps = 0
        if config.add_condition_noise_sampler:
            self.condition_noise_sampler = CategorySpecificConditionNoiseSampler(
                max_state_dim=config.max_state_dim,
                max_action_dim=config.max_action_dim,
                action_horizon=config.action_horizon,
                max_num_embodiments=config.max_num_embodiments,
                num_attention_heads=config.diffusion_model_cfg['num_attention_heads'],
                attention_head_dim=config.diffusion_model_cfg['attention_head_dim'],
                cross_attention_dim=config.backbone_embedding_dim,
                num_layers = 2,
                dropout = config.diffusion_model_cfg['dropout'],
                final_dropout = config.diffusion_model_cfg['final_dropout'],
                min_std = 0.2,
                max_std = 2,
            )
    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample
    def sample_action_noise(self, vl_embeds=None, vl_attn_mask=None, state=None, embodiment_id=None):
        if self.config.add_condition_noise_sampler:
            out_dict = self.condition_noise_sampler(vl_embeds, vl_attn_mask, state, embodiment_id, False, True)
            action_noise = out_dict["noise"]
            std_penalty = out_dict["std_penalty"]
        else:
            action_noise = torch.randn(
                (vl_embeds.shape[0], self.config.action_horizon, self.config.max_action_dim), 
                device=vl_embeds.device, 
                dtype=vl_embeds.dtype
            )
            std_penalty=None
        return action_noise, std_penalty
    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def clean_forward(self, backbone_output: BatchFeature, action_input: BatchFeature, noise=None, t=None, noisy_trajectory=None, velocity=None) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        if noise is None:
            noise, std_penalty = self.sample_action_noise(vl_embeds=vl_embeds, vl_attn_mask=backbone_output.backbone_attention_mask, state=action_input.state, embodiment_id=embodiment_id)
        if t is None:
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        if t.dim() == 1:
            t = t[:, None, None]  # shape (B,1,1) for broadcast
        if noisy_trajectory is None:
            noisy_trajectory = (1 - t) * noise + t * actions
        if velocity is None:
            velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        if self.config.add_condition_noise_sampler:
            loss = loss  + 0.01*std_penalty
        output_dict =  {
            "loss": loss,
            "action_loss": action_loss,
            "pred_velocity": pred_actions,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            "noise": noise,
            "t": t
        }
        return output_dict
    def consistent_forward(self, backbone_output: BatchFeature, action_input: BatchFeature, noise=None, t=None, noisy_trajectory=None, velocity=None, need_repeat=True, consistent_loss_enable=True) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # repeat tensor for consistency leanrning
        backbone_output = copy.copy(backbone_output) #shallow copy
        action_input = copy.copy(action_input) #shallow copy
        if need_repeat:
            for k,v in backbone_output.items():
                if isinstance(backbone_output[k], torch.Tensor):
                    reps = [2] + [1] * (v.dim() - 1)
                    backbone_output[k] = backbone_output[k].repeat(reps)
            for k,v in action_input.items():
                if isinstance(action_input[k], torch.Tensor):
                    reps = [2] + [1] * (v.dim() - 1)
                    action_input[k] = action_input[k].repeat(reps)
        
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        if noise is None:
            # noise, std_penalty = self.sample_action_noise(vl_embeds=vl_embeds, vl_attn_mask=backbone_output.backbone_attention_mask, state=action_input.state, embodiment_id=embodiment_id)
            noise = torch.randn(
                (vl_embeds.shape[0]//2, self.config.action_horizon, self.config.max_action_dim), 
                device=vl_embeds.device, 
                dtype=vl_embeds.dtype
            )
            noise = noise.repeat([2,1,1])
        if t is None:
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        if t.dim() == 1:
            t = t[:, None, None]  # shape (B,1,1) for broadcast
        if noisy_trajectory is None:
            noisy_trajectory = (1 - t) * noise + t * actions
        if velocity is None:
            velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        if consistent_loss_enable:
            # compte consistency loss
            alpha = 0.95
            self.action_loss_ema = alpha*self.action_loss_ema + (1-alpha)*loss.detach()
            pred_actions_1, pred_actions_2 = pred_actions.chunk(2)
            action_mask_1, action_mask_2 = action_mask.chunk(2)
            consistency_loss_1 = F.mse_loss(pred_actions_1, pred_actions_2.detach(), reduction="none") * action_mask_1
            consistency_loss_1 = consistency_loss_1.sum() / (action_mask_1.sum() + 1e-6)
            consistency_loss_2 = F.mse_loss(pred_actions_2, pred_actions_1.detach(), reduction="none") * action_mask_2
            consistency_loss_2 = consistency_loss_2.sum() / (action_mask_2.sum() + 1e-6)
            consistency_loss = (consistency_loss_1 + consistency_loss_2) / 2
            beta = 1/(1+self.action_loss_ema)
            loss += beta*consistency_loss
            self.train_steps += 1
            if self.train_steps % 10 == 0:
                print(f"action_loss_ema: {self.action_loss_ema}, consistency_loss: {consistency_loss}, beta: {beta}")
        else:
            beta = 1/(1+self.action_loss_ema)
            consistency_loss = None
        output_dict =  {
            "loss": loss,
            "consistency_loss": consistency_loss,
            "action_loss": action_loss,
            "pred_velocity": pred_actions,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            "noise": noise,
            "t": t,
            "consistency_loss_beta": beta,
        }
        return output_dict
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        if self.config.use_pgd and self.config.use_consistency_learning and self.training:
            return self.pgd_adv_injector(
                model_forward_fn=self.consistent_forward, 
                backbone_output=backbone_output, 
                action_input=action_input,
                action_noise_sampler=self.sample_action_noise,
            )
        elif self.config.use_pgd and self.training:
            return self.pgd_adv_injector(
                model_forward_fn=self.clean_forward, 
                backbone_output=backbone_output, 
                action_input=action_input,
                action_noise_sampler=self.sample_action_noise,
            )
        elif self.config.use_consistency_learning and self.training:
            output_dict = self.consistent_forward(backbone_output, action_input)
            discard_keys = ["noise", "t","pred_actions"]
            for k in discard_keys:
                if k in output_dict:
                    output_dict.pop(k)
            return output_dict
        else:
            output_dict = self.clean_forward(backbone_output, action_input)
            discard_keys = ["noise", "t","pred_actions"]
            for k in discard_keys:
                if k in output_dict:
                    output_dict.pop(k)
            return output_dict
    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        state: torch.Tensor,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions,_ = self.sample_action_noise(vl_embeds=vl_embeds, vl_attn_mask=backbone_output.backbone_attention_mask, state=state, embodiment_id=embodiment_id)

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            state=action_input.state,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        from gr00t.model.modules.eagle_backbone import EagleBackbone
        return EagleBackbone
    if "InternVL" in config.model_name:
        from gr00t.model.modules.internvl_backbone import InternVLBackbone
        return InternVLBackbone
    if "Qwen" in config.model_name:
        from gr00t.model.modules.qwen_vl_backbone import QwenVLBackbone
        return QwenVLBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
