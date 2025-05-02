import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture


class SoftWeightingHook(MaskingHook):
    """ """
    def __init__(self, num_data, ema_alpha, temp_d, use_joint, device, temp_w, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device
        self.m = ema_alpha
        self.temp_d = temp_d
        self.use_joint = use_joint
        # 
        self.d_score_bank = torch.zeros(num_data).to(device)
        self.o_score_bank = torch.zeros(num_data).to(device)
        self.j_score_ema_t = torch.zeros(num_data).to(device)
        # 
        self.temp_w = temp_w

    @torch.no_grad()
    def get_disagree_scores(self, probs_comm, mode="l1"):
        nb, nh, nc = probs_comm.shape
        if mode == "l1":
            disagree = (probs_comm.unsqueeze(1) - probs_comm.unsqueeze(2)).abs().mean([-3, -2, -1])
            disagree = torch.exp(-disagree / self.temp_d)
        elif mode == "kl":
            marginal_p = probs_comm.mean(dim=0)
            marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # H, H, D, D
            marginal_p = rearrange(marginal_p, "h g d e -> 1 (h g) (d e)")  # 1, (H, H), (D, D)
            pointwise_p = torch.einsum("bhd,bge->bhgde", probs_comm, probs_comm)  # B, H, H, D, D
            pointwise_p = rearrange(pointwise_p, "b h g d e -> b (h g) (d e)")  # B, (H, H), (D, D)
            kl_computed = pointwise_p * (pointwise_p.log() - marginal_p.log())
            kl_grid = rearrange(kl_computed.sum(-1), "b (h g) -> b h g", h=nh)  # B, H, H
            disagree = torch.triu(kl_grid, diagonal=1).mean([-1, -2])
        return disagree

    @torch.no_grad()
    def update_bank(self, algorithm, d_score, o_score, idx):
        if algorithm.distributed and algorithm.world_size > 1:
            d_score = self.concat_all_gather(d_score)
            o_score = self.concat_all_gather(o_score)
        self.d_score_bank[idx] = d_score
        self.o_score_bank[idx] = o_score

    @torch.no_grad()
    def update_score(self, algorithm, j_score, idx):
        if algorithm.distributed and algorithm.world_size > 1:
            j_score = self.concat_all_gather(j_score)
            idx = self.concat_all_gather(idx)
        self.j_score_ema_t[idx] = self.m * self.j_score_ema_t[idx] + (1 - self.m) * j_score

    @torch.no_grad()
    def weighting(self, algorithm, logits_comm, idx, *args, **kwargs):
        probs_comm = torch.softmax(logits_comm, dim=-1)  # B, H, C
        d_score = self.get_disagree_scores(probs_comm)  # B
        o_score = probs_comm.mean(dim=1).max(dim=-1)[0]  # B
        self.update_bank(algorithm, d_score, o_score, idx)
        # 
        _min = self.d_score_bank.min()
        _max = self.d_score_bank.max()
        j_score = ((d_score - _min) / (_max - _min + 1e-8)).abs()
        if self.use_joint:
            _min = self.o_score_bank.min()
            _max = self.o_score_bank.max()
            j_score = j_score * ((o_score - _min) / (_max - _min + 1e-8)).abs()
        # 
        self.update_score(algorithm, j_score, idx)
        return self.j_score_ema_t[idx]

    @torch.no_grad()
    def masking(self, algorithm, idx, *args, **kwargs):
        otsu_th = threshold_otsu(self.j_score_ema_t.cpu().numpy())
        hard_mask = self.j_score_ema_t[idx].ge(otsu_th).to(self.device)
        soft_mask = self.j_score_ema_t[idx] / (otsu_th + 1e-8)
        soft_mask = torch.clamp(soft_mask ** self.temp_w, max=1.0).to(self.device)
        return hard_mask, soft_mask


class DiverseLoss(nn.Module):
    """ """
    def __init__(self, ):
        super().__init__()

    def forward(self, logits):
        num_heads = logits.shape[1]
        probs = torch.softmax(logits, dim=-1) # B, H, D
        # 
        marginal_p = probs.mean(dim=0)  # H, D
        marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # H, H, D, D
        marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # (H, H), (D, D)
        # 
        joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(dim=0)  # H, H, D, D
        joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # (H, H), (D, D)
        # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
        # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
        kl_computed = joint_p * (joint_p.log() - marginal_p.log())
        kl_computed = kl_computed.sum(dim=-1) # (H, H)
        # kl_computed = kl_computed.mean(dim=-1)
        kl_grid = rearrange(kl_computed, "(h g) -> h g", h=num_heads) # H x H
        repulsion_grid = -kl_grid
        # 
        repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
        repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
        repulsion_loss = -repulsions.mean()

        return repulsion_loss
