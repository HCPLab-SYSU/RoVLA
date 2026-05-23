import torch
import torch.nn as nn
import copy
from transformers.feature_extraction_utils import BatchFeature
import torch.nn.functional as F
class PGDAdversarialInjector(nn.Module):
    def __init__(
        self,
        epsilon: float = 0.01,
        alpha: float = 0.01,
        num_steps: int = 3,
        norm: str = "l2",  # "l2" or "linf"
        perturb_backbone_keys: list = ["backbone_features"],   # e.g., ["backbone_features"]
        perturb_action_keys: list = ["state"],     # e.g., ["state"]
        stop_gradient: bool = True,
        w_pertub_loss: float = 0.5,
        perturb_noise: bool = False,
        all_step_loss: bool = False,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.norm = norm
        self.perturb_backbone_keys = set(perturb_backbone_keys or [])
        self.perturb_action_keys = set(perturb_action_keys or [])
        self.stop_gradient = stop_gradient
        self.w_pertub_loss = w_pertub_loss
        self.perturb_noise = perturb_noise
        self.all_step_loss = all_step_loss
        if not self.perturb_backbone_keys and not self.perturb_action_keys:
            raise ValueError("At least one of perturb_backbone_keys or perturb_action_keys must be non-empty.")

    @torch.enable_grad()
    def _pgd_step(self, loss_fn, model_forward_kwargs, action_noise_sampler=None):
        """
        Perform one PGD optimization step on a dict of tensors.
        Only keys in self.perturb_backbone_keys / self.perturb_action_keys are perturbed.
        
        Args:
            model_forward_kwargs: {
                "backbone_output": dict[str, Tensor],
                "action_input": dict[str, Tensor]
            }
            loss_fn: callable that takes (backbone_dict, action_dict) -> dict with 'loss', 'action_loss', 'action_mask', 'backbone_features', 'state_features'
        Returns:
            tuple: (updated_model_forward_kwargs, model_step_outputs,noise, t)
        """
        # Set requires_grad for selected keys
        model_forward_kwargs['backbone_output'] = self._set_key_tensor_to_require_grad(
            model_forward_kwargs["backbone_output"], self.perturb_backbone_keys
        )
        model_forward_kwargs['action_input'] = self._set_key_tensor_to_require_grad(
            model_forward_kwargs["action_input"], self.perturb_action_keys
        ) 
        if self.perturb_noise:
            if model_forward_kwargs.get('noise') is None:
                if action_noise_sampler is not None:
                    model_forward_kwargs['noise'] = action_noise_sampler(
                        model_forward_kwargs['backbone_output'].backbone_features, 
                        model_forward_kwargs['backbone_output'].backbone_attention_mask,
                        model_forward_kwargs['action_input'].state, 
                        model_forward_kwargs['action_input'].embodiment_id
                    ).requires_grad_(True)
                else:
                    model_forward_kwargs['noise'] = torch.randn(
                        model_forward_kwargs["action_input"].action.shape, 
                        device=model_forward_kwargs["action_input"].action.device,
                        dtype=model_forward_kwargs["action_input"].action.dtype
                    ).requires_grad_(True)
            else:
                model_forward_kwargs['noise'] = model_forward_kwargs['noise'].requires_grad_(True)
        noise, t = model_forward_kwargs.get('noise'), model_forward_kwargs.get('t')
        # Compute loss and extract all outputs
        model_step_outputs = loss_fn(model_forward_kwargs["backbone_output"], model_forward_kwargs["action_input"], noise, t)
        model_forward_kwargs['noise'] = model_step_outputs['noise']
        model_forward_kwargs['t'] = model_step_outputs['t']
        loss = model_step_outputs['loss']
        
        # Collect tensors that actually require gradient (only perturbable keys)
        backbone_tensors = []
        backbone_keys_ordered = []
        for k in self.perturb_backbone_keys:
            tensor = model_forward_kwargs["backbone_output"][k]
            backbone_tensors.append(tensor)
            backbone_keys_ordered.append(k)

        action_tensors = []
        action_keys_ordered = []
        for k in self.perturb_action_keys:
            tensor = model_forward_kwargs["action_input"][k]
            action_tensors.append(tensor)
            action_keys_ordered.append(k)

        all_tensors = backbone_tensors + action_tensors
        if self.perturb_noise:
            all_tensors.append(noise)
        # Compute gradients only w.r.t. the perturbable tensors
        grads = torch.autograd.grad(loss, all_tensors, retain_graph=True)

        # Split grads
        n_b = len(backbone_tensors)
        n_a = len(action_tensors)
        backbone_grads = grads[:n_b]
        action_grads = grads[n_b:n_b+n_a]
        if self.perturb_noise:
            noise_grad = grads[-1]
            # Update noise if needed
            noise_next = self._project_update(noise, noise_grad, self.epsilon, self.alpha)

        # Build updated dict: start with original, then replace perturbed keys
        model_forward_kwargs_next = {
            "backbone_output": copy.copy(model_forward_kwargs["backbone_output"]),  # shallow copy
            "action_input": copy.copy(model_forward_kwargs["action_input"]),
            "noise": noise_next if self.perturb_noise else model_forward_kwargs.get('noise'),
            "t": model_forward_kwargs.get('t'),
        }

        # Update backbone perturbable keys
        for (k, g) in zip(backbone_keys_ordered, backbone_grads):
            x = model_forward_kwargs["backbone_output"][k]
            model_forward_kwargs_next["backbone_output"][k] = self._project_update(x, g, self.epsilon, self.alpha)

        # Update action perturbable keys
        for (k, g) in zip(action_keys_ordered, action_grads):
            x = model_forward_kwargs["action_input"][k]
            model_forward_kwargs_next["action_input"][k] = self._project_update(x, g, self.epsilon, self.alpha)
        return model_forward_kwargs, model_forward_kwargs_next, model_step_outputs

    def _project_update(self, x_clean, grad, eps, step_size):
        if self.norm == "linf":
            x_new = x_clean + step_size * grad.sign()
            x_new = torch.clamp(x_new, x_clean - eps, x_clean + eps)
        elif self.norm == "l2":
            # 计算 x_clean 的 L2 范数（用于相对扰动）
            x_norm = torch.norm(x_clean.view(x_clean.shape[0], -1), p=2, dim=1, keepdim=True)
            x_norm = x_norm.view(-1, *[1] * (x_clean.dim() - 1))
            
            # 避免除零（如果 x_clean 全为 0，则退化为绝对 eps）
            x_norm_safe = torch.where(x_norm < 1e-8, torch.ones_like(x_norm), x_norm)
            
            # 相对扰动上限：eps_rel * ||x_clean||
            eps_relative = eps * x_norm_safe

            # 归一化梯度
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
            grad = grad / (grad_norm + 1e-8)

            # 初步更新
            x_new = x_clean + step_size * grad
            diff = x_new - x_clean

            # 计算实际扰动范数
            diff_norm = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=1, keepdim=True)
            diff_norm = diff_norm.view(-1, *[1] * (diff.dim() - 1))

            # 投影到相对 eps 球内
            factor = torch.min(eps_relative / (diff_norm + 1e-8), torch.ones_like(diff_norm))
            diff = diff * factor
            # print('diff:',diff.abs().mean()/x_clean.abs().mean())
            x_new = x_clean + diff.detach()
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")
        return x_new.requires_grad_(True)

    def _set_key_tensor_to_require_grad(self, batch_feature, keys):
        """Set selected keys to require grad."""
        for k in keys:
            batch_feature[k] = batch_feature[k].requires_grad_(True)
        return batch_feature

    def forward(self, model_forward_fn, backbone_output, action_input, action_noise_sampler=None):
        """
        Args:
            backbone_output: BatchFeature
            action_input: BatchFeature
            model_forward_fn: callable(backbone, action) -> {"loss": Tensor}
        Returns:
            {"loss": Tensor}
        """
        if not self.training:
            # 在非训练模式下，直接调用model_forward_fn获取完整输出
            return model_forward_fn(backbone_output, action_input)
        backbone_output, action_input = copy.copy(backbone_output), copy.copy(action_input) # shallow copy
        model_forward_kwargs = {
            "backbone_output": backbone_output,
            "action_input": action_input,
            "noise": None,
            "t": None,
        }
        clean_outputs = None
        step_loss_list = []
        step_action_loss_list = []
        for _ in range(self.num_steps):
            model_forward_kwargs, model_forward_kwargs_next, model_step_outputs = self._pgd_step(model_forward_fn, model_forward_kwargs, action_noise_sampler)
            step_loss_list.append(model_step_outputs['loss'])
            step_action_loss_list.append(model_step_outputs.get('action_loss'))
            if clean_outputs is None:
                clean_outputs = model_step_outputs
            if self.stop_gradient:
                # Detach all perturbed tensors
                for k in self.perturb_backbone_keys:
                    model_forward_kwargs_next["backbone_output"][k] = model_forward_kwargs_next["backbone_output"][k].detach()
                for k in self.perturb_action_keys:
                    model_forward_kwargs_next["action_input"][k] = model_forward_kwargs_next["action_input"][k].detach()
                if self.noise_perturb:
                    model_forward_kwargs_next['noise'] = model_forward_kwargs_next['noise'].detach()
            model_forward_kwargs = copy.copy(model_forward_kwargs_next)#shallow copy
        final_perturbed_outputs = model_forward_fn(**model_forward_kwargs_next)
        step_loss_list.append(final_perturbed_outputs['loss']) # include final step loss
        step_action_loss_list.append(final_perturbed_outputs.get('action_loss'))# include final step action loss

        clean_loss = step_loss_list[0] # first step loss as clean loss
        clean_action_loss = step_action_loss_list[0]
        if self.all_step_loss:
            perturbed_loss = sum(step_loss_list[1:]) / (len(step_loss_list)-1)
            if step_action_loss_list[-1] is not None:
                perturbed_action_loss = sum(step_action_loss_list[1:]) / (len(step_action_loss_list)-1)
            else:
                perturbed_action_loss = None
        else:
            perturbed_loss = step_loss_list[-1]
            perturbed_action_loss = step_action_loss_list[-1]
        weighted_loss = (1-self.w_pertub_loss)*clean_loss + self.w_pertub_loss*perturbed_loss
        # 对action_loss也进行加权组合
        if clean_action_loss is not None and perturbed_action_loss is not None:
            weighted_action_loss = (1-self.w_pertub_loss)*clean_action_loss + self.w_pertub_loss*perturbed_action_loss
        else:
            weighted_action_loss = None
            
        # 返回包含所有必要字段的完整输出
        output_dict = {
            "loss": weighted_loss,
            "action_loss": weighted_action_loss,
            "action_mask": clean_outputs.get('action_mask'),
            "backbone_features": clean_outputs.get("backbone_features"),
            "state_features": clean_outputs.get("state_features"),
        }
        return output_dict

class CSPGDAdversarialInjector(nn.Module):
    def __init__(
        self,
        epsilon: float = 0.01,
        alpha: float = 0.01,
        num_steps: int = 3,
        norm: str = "l2",  # "l2" or "linf"
        perturb_backbone_keys: list = ["backbone_features"],   # e.g., ["backbone_features"]
        perturb_action_keys: list = ["state"],     # e.g., ["state"]
        stop_gradient: bool = True,
        w_pertub_loss: float = 0.5,
        perturb_noise: bool = False,
        all_step_loss: bool = False,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.norm = norm
        self.perturb_backbone_keys = set(perturb_backbone_keys or [])
        self.perturb_action_keys = set(perturb_action_keys or [])
        self.stop_gradient = stop_gradient
        self.w_pertub_loss = w_pertub_loss
        self.perturb_noise = perturb_noise
        self.all_step_loss = all_step_loss
        if not self.perturb_backbone_keys and not self.perturb_action_keys:
            raise ValueError("At least one of perturb_backbone_keys or perturb_action_keys must be non-empty.")

    @torch.enable_grad()
    def _pgd_step(self, loss_fn, model_forward_kwargs, action_noise_sampler=None):
        """
        Perform one PGD optimization step on a dict of tensors.
        Only keys in self.perturb_backbone_keys / self.perturb_action_keys are perturbed.
        
        Args:
            model_forward_kwargs: {
                "backbone_output": dict[str, Tensor],
                "action_input": dict[str, Tensor]
            }
            loss_fn: callable that takes (backbone_dict, action_dict) -> dict with 'loss', 'action_loss', 'action_mask', 'backbone_features', 'state_features'
        Returns:
            tuple: (updated_model_forward_kwargs, model_step_outputs,noise, t)
        """
        # Set requires_grad for selected keys
        model_forward_kwargs['backbone_output'] = self._set_key_tensor_to_require_grad(
            model_forward_kwargs["backbone_output"], self.perturb_backbone_keys
        )
        model_forward_kwargs['action_input'] = self._set_key_tensor_to_require_grad(
            model_forward_kwargs["action_input"], self.perturb_action_keys
        ) 
        if self.perturb_noise:
            if model_forward_kwargs.get('noise') is None:
                if action_noise_sampler is not None:
                    model_forward_kwargs['noise'] = action_noise_sampler(
                        model_forward_kwargs['backbone_output'].backbone_features, 
                        model_forward_kwargs['backbone_output'].backbone_attention_mask,
                        model_forward_kwargs['action_input'].state, 
                        model_forward_kwargs['action_input'].embodiment_id
                    ).requires_grad_(True)
                else:
                    model_forward_kwargs['noise'] = torch.randn(
                        model_forward_kwargs["action_input"].action.shape, 
                        device=model_forward_kwargs["action_input"].action.device,
                        dtype=model_forward_kwargs["action_input"].action.dtype
                    ).requires_grad_(True)
            else:
                model_forward_kwargs['noise'] = model_forward_kwargs['noise'].requires_grad_(True)
        noise, t = model_forward_kwargs.get('noise'), model_forward_kwargs.get('t')
        # Compute loss and extract all outputs
        model_step_outputs = loss_fn(model_forward_kwargs["backbone_output"], model_forward_kwargs["action_input"], noise, t)
        model_forward_kwargs['noise'] = model_step_outputs['noise']
        model_forward_kwargs['t'] = model_step_outputs['t']
        consistency_loss = model_step_outputs['consistency_loss']
        
        # Collect tensors that actually require gradient (only perturbable keys)
        backbone_tensors = []
        backbone_keys_ordered = []
        for k in self.perturb_backbone_keys:
            tensor = model_forward_kwargs["backbone_output"][k]
            backbone_tensors.append(tensor)
            backbone_keys_ordered.append(k)

        action_tensors = []
        action_keys_ordered = []
        for k in self.perturb_action_keys:
            tensor = model_forward_kwargs["action_input"][k]
            action_tensors.append(tensor)
            action_keys_ordered.append(k)

        all_tensors = backbone_tensors + action_tensors
        if self.perturb_noise:
            all_tensors.append(noise)
        # Compute gradients only w.r.t. the perturbable tensors
        grads = torch.autograd.grad(consistency_loss, all_tensors, retain_graph=True)

        # Split grads
        n_b = len(backbone_tensors)
        n_a = len(action_tensors)
        backbone_grads = grads[:n_b]
        action_grads = grads[n_b:n_b+n_a]
        if self.perturb_noise:
            noise_grad = grads[-1]
            # Update noise if needed
            noise_next = self._project_update(noise, noise_grad, self.epsilon, self.alpha)

        # Build updated dict: start with original, then replace perturbed keys
        model_forward_kwargs_next = {
            "backbone_output": copy.copy(model_forward_kwargs["backbone_output"]),  # shallow copy
            "action_input": copy.copy(model_forward_kwargs["action_input"]),
            "noise": noise_next if self.perturb_noise else model_forward_kwargs.get('noise'),
            "t": model_forward_kwargs.get('t'),
        }

        # Update backbone perturbable keys
        for (k, g) in zip(backbone_keys_ordered, backbone_grads):
            x = model_forward_kwargs["backbone_output"][k]
            model_forward_kwargs_next["backbone_output"][k] = self._project_update(x, g, self.epsilon, self.alpha, x_mask=model_forward_kwargs["backbone_output"].image_mask.unsqueeze(-1))

        # Update action perturbable keys
        for (k, g) in zip(action_keys_ordered, action_grads):
            x = model_forward_kwargs["action_input"][k]
            model_forward_kwargs_next["action_input"][k] = self._project_update(x, g, self.epsilon, self.alpha)
        return model_forward_kwargs, model_forward_kwargs_next, model_step_outputs

    def _project_update(self, x_clean, grad, eps, step_size, x_mask=None):
        if self.norm == "linf":
            diff = step_size * grad.sign()
            if x_mask is not None:
                diff = diff * x_mask.to(diff.dtype)
            x_new = x_clean + diff
            x_new = torch.clamp(x_new, x_clean - eps, x_clean + eps)
        elif self.norm == "l2":
            # 计算 x_clean 的 L2 范数（用于相对扰动）
            x_norm = torch.norm(x_clean.view(x_clean.shape[0], -1), p=2, dim=1, keepdim=True)
            x_norm = x_norm.view(-1, *[1] * (x_clean.dim() - 1))
            
            # 避免除零（如果 x_clean 全为 0，则退化为绝对 eps）
            x_norm_safe = torch.where(x_norm < 1e-8, torch.ones_like(x_norm), x_norm)
            
            # 相对扰动上限：eps_rel * ||x_clean||
            eps_relative = eps * x_norm_safe

            # 归一化梯度
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
            grad = grad / (grad_norm + 1e-8)

            # 初步更新
            x_new = x_clean + step_size * grad
            diff = x_new - x_clean

            # 计算实际扰动范数
            diff_norm = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=1, keepdim=True)
            diff_norm = diff_norm.view(-1, *[1] * (diff.dim() - 1))

            # 投影到相对 eps 球内
            factor = torch.min(eps_relative / (diff_norm + 1e-8), torch.ones_like(diff_norm))
            diff = diff * factor
            if x_mask is not None:
                diff = diff * x_mask.to(diff.dtype)
            # print('diff:',diff.abs().mean()/x_clean.abs().mean())
            x_new = x_clean + diff.detach()
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")
        return x_new.requires_grad_(True)

    def _set_key_tensor_to_require_grad(self, batch_feature, keys):
        """Set selected keys to require grad."""
        for k in keys:
            batch_feature[k] = batch_feature[k].requires_grad_(True)
        return batch_feature

    def forward(self, model_forward_fn, backbone_output, action_input, action_noise_sampler=None):
        """
        Args:
            backbone_output: BatchFeature
            action_input: BatchFeature
            model_forward_fn: callable(backbone, action) -> {"loss": Tensor}
        Returns:
            {"loss": Tensor}
        """
        if not self.training:
            # 在非训练模式下，直接调用model_forward_fn获取完整输出
            return model_forward_fn(backbone_output, action_input)
        backbone_output, action_input = copy.copy(backbone_output), copy.copy(action_input) # shallow copy
        model_forward_kwargs = {
            "backbone_output": backbone_output,
            "action_input": action_input,
            "noise": None,
            "t": None,
            "noisy_trajectory":None, 
            "velocity":None, 
            "need_repeat":True, 
            "consistent_loss_enable":True
        }
        clean_outputs = None
        step_loss_list = []
        step_action_loss_list = []
        for step in range(self.num_steps):
            model_forward_kwargs, model_forward_kwargs_next, model_step_outputs = self._pgd_step(model_forward_fn, model_forward_kwargs, action_noise_sampler)
            step_loss = model_step_outputs['loss']
            if step==0:
                clean_outputs = model_step_outputs
            else:
                # step perturbed -> clean consistency loss
                assert not model_forward_kwargs['consistent_loss_enable'], "consistency loss should be disabled when step > 0"
                step_consistency_loss = F.mse_loss(model_step_outputs['pred_velocity'], clean_outputs['pred_velocity'].detach(), reduction="none") * clean_outputs['action_mask']
                step_consistency_loss = step_consistency_loss.sum()/(clean_outputs['action_mask'].sum()+1e-6)
                step_loss = step_loss + clean_outputs['consistency_loss_beta']*step_consistency_loss
            model_forward_kwargs_next['consistent_loss_enable']=False
            step_loss_list.append(step_loss)
            step_action_loss_list.append(model_step_outputs.get('action_loss'))
            if self.stop_gradient:
                # Detach all perturbed tensors
                for k in self.perturb_backbone_keys:
                    model_forward_kwargs_next["backbone_output"][k] = model_forward_kwargs_next["backbone_output"][k].detach()
                for k in self.perturb_action_keys:
                    model_forward_kwargs_next["action_input"][k] = model_forward_kwargs_next["action_input"][k].detach()
                if self.noise_perturb:
                    model_forward_kwargs_next['noise'] = model_forward_kwargs_next['noise'].detach()
            model_forward_kwargs = copy.copy(model_forward_kwargs_next) # shallow copy
        final_perturbed_outputs = model_forward_fn(**model_forward_kwargs_next)
        
        # perturbed -> clean consistency loss
        final_consistency_loss = F.mse_loss(final_perturbed_outputs['pred_velocity'], clean_outputs['pred_velocity'].detach(), reduction="none") * clean_outputs['action_mask']
        final_consistency_loss = final_consistency_loss.sum()/(clean_outputs['action_mask'].sum()+1e-6)
        final_perturbed_loss = final_perturbed_outputs['loss'] + clean_outputs['consistency_loss_beta']*final_consistency_loss
        
        step_loss_list.append(final_perturbed_loss) # include final step loss
        step_action_loss_list.append(final_perturbed_outputs.get('action_loss'))# include final step action loss
        
        
        clean_loss = step_loss_list[0] # first step loss as clean loss
        clean_action_loss = step_action_loss_list[0]
        if self.all_step_loss:
            perturbed_loss = sum(step_loss_list[1:]) / (len(step_loss_list)-1)
            if step_action_loss_list[-1] is not None:
                perturbed_action_loss = sum(step_action_loss_list[1:]) / (len(step_action_loss_list)-1)
            else:
                perturbed_action_loss = None
        else:
            perturbed_loss = step_loss_list[-1]
            perturbed_action_loss = step_action_loss_list[-1]
        weighted_loss = (1-self.w_pertub_loss)*clean_loss + self.w_pertub_loss*perturbed_loss
        # 对action_loss也进行加权组合
        if clean_action_loss is not None and perturbed_action_loss is not None:
            weighted_action_loss = (1-self.w_pertub_loss)*clean_action_loss + self.w_pertub_loss*perturbed_action_loss
        else:
            weighted_action_loss = None
            
        # 返回包含所有必要字段的完整输出
        output_dict = {
            "loss": weighted_loss,
            "action_loss": weighted_action_loss,
            "action_mask": clean_outputs.get('action_mask'),
            "backbone_features": clean_outputs.get("backbone_features"),
            "state_features": clean_outputs.get("state_features"),
        }
        return output_dict

if __name__ == "__main__":
    # -----------------------------
    # 测试用例：包含 backbone 网络
    # -----------------------------
    class Backbone(nn.Module):
        def __init__(self, input_dim=128, feat_dim=64):
            super().__init__()
            self.linear = nn.Linear(input_dim, feat_dim)

        def forward(self, obs):
            return self.linear(obs)  # (B, feat_dim)

    class PolicyHead(nn.Module):
        def __init__(self, feat_dim=64, action_dim=10):
            super().__init__()
            self.head = nn.Linear(feat_dim, action_dim)
            self.pgd = PGDmodel_forward_kwargsersarialInjector(
                epsilon=0.1,
                alpha=0.05,
                num_steps=1,
                norm="l2",
                perturb_backbone_keys=["backbone_features"],
                perturb_action_keys=["state"],
                stop_gradient=False,  # 设为 False 以验证端到端梯度
                w_pertub_loss=1.0,
            )
        def clean_forward(self, backbone_features, state):
            # 简单融合
            x = backbone_features['backbone_features'] + state['state']
            x = self.head(x)
            target = torch.randint(0, x.shape[-1], (B,), device=device)
            loss = nn.CrossEntropyLoss()(x, target)
            return {"loss": loss}
        def forward(self, backbone_features, state):
            return self.pgd(self.clean_forward, {'backbone_features':backbone_features}, {"state":state})
    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.policy = PolicyHead()

        def forward(self, obs, state):
            backbone_features = self.backbone(obs)  # 这里保留梯度！
            loss = self.policy(backbone_features, state)['loss']
            return loss
    
    device = "cpu"
    B, obs_dim, feat_dim, act_dim = 4, 128, 64, 10

    model = FullModel().to(device)

    model.train()
    # 原始输入
    obs = torch.randn(B, obs_dim, device=device)
    state = torch.randn(B, feat_dim, device=device)

    # 前向：获取 backbone_features（带梯度！）
    total_loss = model(obs, state)

    # print(f"Total loss: {total_loss.item():.4f}")

    # 验证 backbone 参数是否收到梯度
    total_loss.backward()

    # 检查 backbone.linear.weight 是否有梯度
    grad_norm = model.backbone.linear.weight.grad.norm().item()
    # grad_norm = model.policy.head.weight.grad.norm().item()
    # print(f"Backbone weight grad norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Backbone should receive gradient from model_forward_kwargsersarial loss!"

    # print("✅ End-to-end model_forward_kwargsersarial training works!")