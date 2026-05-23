import torch
def jerk_score(actions, action_mask, dt=0.1):
    """
    actions : Tensor, (B, H, D)  已归一化到合理区间
    action_mask: Tensor, (B, H)  动作是否有效
    dt      : 控制周期，默认 0.1 s
    return_per_step: 是否额外返回 (B, H-3) 的逐帧 jerk
    return  : Tensor, (B,)  越大越平滑
    """
    B, H, D = actions.shape
    if H < 3:
        # 轨迹太短，直接给零分
        return torch.zeros(B, dtype=actions.dtype, device=actions.device)

    # 将无效动作置为0，避免影响差分计算
    masked_actions = actions * action_mask.unsqueeze(-1)
    
    # 1. 一阶差分 → 速度  (B, H-1, D)
    vel = masked_actions[:, 1:] - masked_actions[:, :-1]
    vel = vel / dt

    # 2. 二阶差分 → 加速度  (B, H-2, D)
    acc = vel[:, 1:] - vel[:, :-1]
    acc = acc / dt

    # 3. 三阶差分 → jerk    (B, H-3, D)
    jerk = acc[:, 1:] - acc[:, :-1]
    jerk = jerk / dt        # 单位统一

    # 4. 把 D 维 jerk 大小合并：L2 范数  (B, H-3)
    jerk_norm = torch.norm(jerk, p=2, dim=-1)

    # 5. 使用 action_mask 计算有效的 jerk 平均值
    # jerk 对应的时间步是 [1:H-1]，所以对应的 mask 是 action_mask[:, 1:H-1]
    valid_mask = action_mask[:, 3:]  # (B, H-3)
    
    # 将无效时间步的 jerk 置为0
    jerk_norm = jerk_norm * valid_mask
    
    # 计算有效时间步的平均 jerk
    valid_count = valid_mask.sum(dim=1)  # (B,)
    valid_count = torch.clamp(valid_count, min=1)  # 避免除零
    mean_jerk = jerk_norm.sum(dim=1) / valid_count

    # 6. 转成 0~1 的分数，越大越平滑
    score = 1.0 / (1.0 + mean_jerk)
    return score