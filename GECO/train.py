import os

import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# -----------------------------
# 三层残差 MLP
# -----------------------------
class ResidualMLP3(nn.Module):
    """
    三层 MLP（两段残差）结构：
        x -> proj_in -> h
        h -> (LN->Act->Linear->Dropout) + h -> y
        y -> (LN->Act->Linear->Dropout) + y -> t
        t -> (可选 LN) -> Linear -> out
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        final_norm: bool = True,
        act: str = "gelu",
    ):
        super().__init__()
        self.final_norm = final_norm
        self.proj_in = nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.ff1 = nn.Linear(hidden_dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(hidden_dim, out_dim)

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        y = h + self.dropout(self.ff1(self.act(self.norm1(h))))
        t = y + self.dropout(self.ff2(self.act(self.norm2(y))))
        if self.final_norm:
            t = self.norm_out(t)
        return self.out(t)


# -----------------------------
# Dataset：LLaVA/Qwen 单token + CLIP 序列(768) + Gemma 单token(2560)
# -----------------------------
class FourPlayerDataset(Dataset):
    """
    期望 json 里每条包含（注意：json.load 要求整体是 JSON 数组，不是 jsonl）：
      - 'hidden_state_file'   : LLaVA (.pt)
      - 'hidden_state_file2'  : Qwen  (.pt)
      - 'hidden_state_file3'  : CLIP 图像序列 (.pt) [Li,768] or [768]
      - 'hidden_state_file4'  : CLIP 文本序列 (.pt) [Lt,768] or [768]
      - 'hidden_state_file5'  : Gemma (.pt) [*,2560]（取最后一个 token）
      - 'label'               : 0/1
    """
    def __init__(self, json_file: str, data_dir: str, max_seq_img: int = 256, max_seq_txt: int = 256, clip_dim: int = 512, preload: bool = True):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)  # 必须是 JSON 数组
        self.data_dir = Path(data_dir)
        self.max_seq_img = max_seq_img
        self.max_seq_txt = max_seq_txt
        self.clip_dim = clip_dim
        self.preload = preload

        self.cache = []
        if self.preload:
            print(f"⚡ Preloading {len(self.data)} feature files into RAM for 20x speedup...")
            for idx in range(len(self.data)):
                self.cache.append(self._load_item(idx))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _last_token(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.float()
        if x.dim() == 2:
            return x[-1].float()
        raise ValueError(f"Unexpected tensor dim {x.dim()} for single-token backbone.")

    def _as_seq(self, x: torch.Tensor, max_len: int, expect_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.size(-1) != expect_dim:
            raise ValueError(f"Expect (seq,{expect_dim}) or ({expect_dim},), got {tuple(x.shape)}")
        if x.size(0) > max_len:
            x = x[:max_len]
        mask = torch.ones(x.size(0), dtype=torch.bool)
        return x.float(), mask

    def _load_item(self, idx):
        item = self.data[idx]
        llava = torch.load(self.data_dir / item["hidden_state_file"], map_location="cpu")
        qwen = torch.load(self.data_dir / item["hidden_state_file2"], map_location="cpu")
        clip_img = torch.load(self.data_dir / item["hidden_state_file3"], map_location="cpu")
        clip_txt = torch.load(self.data_dir / item["hidden_state_file4"], map_location="cpu")
        gemma = torch.load(self.data_dir / item["hidden_state_file5"], map_location="cpu")

        llava_tok = self._last_token(llava)
        qwen_tok = self._last_token(qwen)
        gemma_tok = self._last_token(gemma)

        clip_img_seq, clip_img_mask = self._as_seq(clip_img, self.max_seq_img, self.clip_dim)
        clip_txt_seq, clip_txt_mask = self._as_seq(clip_txt, self.max_seq_txt, self.clip_dim)

        return {
            "llava": llava_tok,
            "qwen": qwen_tok,
            "gemma": gemma_tok,
            "clip_img_seq": clip_img_seq,
            "clip_txt_seq": clip_txt_seq,
            "clip_img_mask": clip_img_mask,
            "clip_txt_mask": clip_txt_mask,
            "label": int(item["label"]),
            "id": item.get("id", str(idx)),
        }

    def __getitem__(self, idx):
        if self.preload:
            return self.cache[idx]
        return self._load_item(idx)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    llava = torch.stack([b["llava"] for b in batch])
    qwen = torch.stack([b["qwen"] for b in batch])
    gemma = torch.stack([b["gemma"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    img_seqs = [b["clip_img_seq"] for b in batch]
    txt_seqs = [b["clip_txt_seq"] for b in batch]
    img_masks = [b["clip_img_mask"] for b in batch]
    txt_masks = [b["clip_txt_mask"] for b in batch]

    img_pad = pad_sequence(img_seqs, batch_first=True, padding_value=0.0)
    txt_pad = pad_sequence(txt_seqs, batch_first=True, padding_value=0.0)
    img_msk = pad_sequence(img_masks, batch_first=True, padding_value=False)
    txt_msk = pad_sequence(txt_masks, batch_first=True, padding_value=False)

    return {
        "llava": llava,
        "qwen": qwen,
        "gemma": gemma,
        "clip_img_seq": img_pad,
        "clip_txt_seq": txt_pad,
        "clip_img_mask": img_msk,
        "clip_txt_mask": txt_msk,
        "label": labels,
    }


# -----------------------------
# CLIP 序列编码器（融合 img/text，key-less attention）
# -----------------------------
class CLIPSeqEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 4,
        nhead: int = 8,
        ff_dim: int = 4096,
        dropout: float = 0.1,
        max_img_len: int = 577,
        max_txt_len: int = 77,
        out_proj_dim: int = 768,
    ):
        super().__init__()
        self.hidden = hidden_dim
        self.max_img_len = max_img_len
        self.max_txt_len = max_txt_len
        self.max_total = max_img_len + max_txt_len

        self.pos_embed = nn.Parameter(torch.randn(1, self.max_total, hidden_dim) * 0.02)
        self.seg_embed = nn.Embedding(2, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(hidden_dim)

        self.fuse_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        self.out_proj = ResidualMLP3(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_proj_dim,
            dropout=dropout,
            final_norm=True,
            act="gelu",
        )

    def forward(
        self,
        img_seq: torch.Tensor,
        txt_seq: torch.Tensor,
        img_mask: torch.Tensor,
        txt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        img_seq : (B, Li, 768) —— 约定第 0 位为 v_cls
        txt_seq : (B, Lt, 768) —— 约定最后一个有效 token 为 t_cls（EOT/CLS）
        mask: True 表示有效
        """
        B, Li, _ = img_seq.shape
        _, Lt, _ = txt_seq.shape
        if Li > self.max_img_len or Lt > self.max_txt_len:
            raise ValueError(f"Li/Lt exceed max: Li={Li} (max {self.max_img_len}), Lt={Lt} (max {self.max_txt_len})")

        x = torch.cat([img_seq, txt_seq], dim=1)               # (B, L, D)
        mask_valid = torch.cat([img_mask, txt_mask], dim=1)    # (B, L)
        L = x.size(1)

        seg = torch.cat(
            [
                torch.zeros(B, Li, dtype=torch.long, device=x.device),
                torch.ones(B, Lt, dtype=torch.long, device=x.device),
            ],
            dim=1,
        )

        pos = self.pos_embed[:, :L, :]
        x = x + pos + self.seg_embed(seg)

        key_padding_mask = ~mask_valid  # True=ignore
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.enc_norm(x)

        v_cls = x[:, 0, :]  # (B,D)
        txt_len = txt_mask.sum(dim=1)  # (B,)
        t_cls_idx = Li + (txt_len - 1)  # (B,)
        batch_idx = torch.arange(B, device=x.device)
        t_cls = x[batch_idx, t_cls_idx, :]

        score = self.fuse_score(torch.cat([t_cls, v_cls], dim=-1))  # (B,2)
        pv = F.softmax(score, dim=-1)                               # [p_t, p_v]
        p_t = pv[:, 0].unsqueeze(-1)
        p_v = pv[:, 1].unsqueeze(-1)

        fused = p_t * t_cls + p_v * v_cls
        clip_feat = self.out_proj(fused)

        debug = {
            "gate_mean": p_v.squeeze(-1),   # (B,)
            "keyless_alpha2": pv,           # (B,2)
        }
        return clip_feat, debug


# -----------------------------
# 四分支投影 & 策略头 & 最终分类器策略头（pi_F）
# -----------------------------
class FourBackbones(nn.Module):
    def __init__(
        self,
        clip_dim=512,
        proj_dim=768,
        clip_out_dim=768,
        llava_dim=4096,
        qwen_dim=3584,
        gemma_dim=2560,
        mlp_dropout=0.1,
        max_img_len=577,
        max_txt_len=77,
    ):
        super().__init__()
        self.llava_proj = ResidualMLP3(llava_dim, proj_dim, proj_dim, dropout=mlp_dropout, final_norm=True, act="gelu")
        self.qwen_proj = ResidualMLP3(qwen_dim, proj_dim, proj_dim, dropout=mlp_dropout, final_norm=True, act="gelu")
        self.gemma_proj = ResidualMLP3(gemma_dim, proj_dim, proj_dim, dropout=mlp_dropout, final_norm=True, act="gelu")

        self.clip_enc = CLIPSeqEncoder(
            hidden_dim=clip_dim,
            num_layers=4,
            nhead=8,
            ff_dim=4096,
            dropout=mlp_dropout,
            max_img_len=max_img_len,
            max_txt_len=max_txt_len,
            out_proj_dim=clip_out_dim,
        )

        # 五个玩家策略头（2动作：0/1）
        self.pi_L = ResidualMLP3(proj_dim, 512, 2, dropout=mlp_dropout, final_norm=False, act="gelu")
        self.pi_Q = ResidualMLP3(proj_dim, 512, 2, dropout=mlp_dropout, final_norm=False, act="gelu")
        self.pi_C = ResidualMLP3(clip_out_dim, 512, 2, dropout=mlp_dropout, final_norm=False, act="gelu")
        self.pi_G = ResidualMLP3(proj_dim, 512, 2, dropout=mlp_dropout, final_norm=False, act="gelu")

        fused_dim = proj_dim * 3 + clip_out_dim

        # 仅用于 kl_ref="teacher"（如果你不开 teacher，可以删掉 clf）
        self.clf = ResidualMLP3(fused_dim, 512, 2, dropout=mlp_dropout, final_norm=False, act="gelu")

        # ✅ 最终分类器策略（参与博弈 & 最终 logits）
        self.pi_F = ResidualMLP3(fused_dim, 512, 2, dropout=mlp_dropout, final_norm=False, act="gelu")

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        llava = self.llava_proj(batch["llava"])
        qwen = self.qwen_proj(batch["qwen"])
        gemma = self.gemma_proj(batch["gemma"])

        clip, clip_dbg = self.clip_enc(
            batch["clip_img_seq"],
            batch["clip_txt_seq"],
            batch["clip_img_mask"],
            batch["clip_txt_mask"],
        )

        logit_L = self.pi_L(llava)
        logit_Q = self.pi_Q(qwen)
        logit_C = self.pi_C(clip)
        logit_G = self.pi_G(gemma)

        fused = torch.cat([llava.detach(), qwen.detach(), gemma.detach(), clip.detach()], dim=-1)

        logit_cls = self.clf(fused)    # teacher 参考用（不参与损失）
        logit_F = self.pi_F(fused)     # 最终 logits / pi_F

        return {
            "llava": llava,
            "qwen": qwen,
            "gemma": gemma,
            "clip": clip,
            "pi_L": logit_L,
            "pi_Q": logit_Q,
            "pi_C": logit_C,
            "pi_G": logit_G,
            "pi_F": logit_F,
            "logits": logit_F,
            "clip_dbg": clip_dbg,
            "logits_clf": logit_cls,
        }


# -----------------------------
# 五人版 NAL 线性目标（两两奖励 + 全员全对奖励）
# -----------------------------
def nash_advantage_linear_nal5(
    piL: torch.Tensor,
    piQ: torch.Tensor,
    piC: torch.Tensor,
    piG: torch.Tensor,
    piF: torch.Tensor,
    labels: torch.Tensor,
    acc_reward=(1.0, 1.0, 1.0, 1.0, 1.0),
    coop_bonus: float = 0.2,
    tau_reg: float = 0.0,
    xhat_type: str = "uniform",  # "current" | "uniform" | "custom"
    xhat_custom: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    loss_weight=(1.0, 1.0, 1.0, 1.0, 1.0),
    pair_bonus: float = 0.1,
    pair_bonus_player: Optional[Tuple[float, float, float, float, float]] = None,
    detach_F_in_game: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """
    收益结构（以玩家 i 为例）：
      u_i = α_i·1[a_i=y]
            + λ_i·sum_{j≠i} 1[a_i=y, a_j=y]
            + β·1[all five correct]
    """
    piL = piL.clamp(min=1e-7, max=1.0)
    piQ = piQ.clamp(min=1e-7, max=1.0)
    piC = piC.clamp(min=1e-7, max=1.0)
    piG = piG.clamp(min=1e-7, max=1.0)
    piF = piF.clamp(min=1e-7, max=1.0)

    B, dev = labels.size(0), labels.device
    y = labels.view(B, 1, 1, 1, 1, 1)
    a2 = torch.arange(2, device=dev)

    corr_L = (a2.view(1, 2, 1, 1, 1, 1) == y).float()
    corr_Q = (a2.view(1, 1, 2, 1, 1, 1) == y).float()
    corr_C = (a2.view(1, 1, 1, 2, 1, 1) == y).float()
    corr_G = (a2.view(1, 1, 1, 1, 2, 1) == y).float()
    corr_F = (a2.view(1, 1, 1, 1, 1, 2) == y).float()

    all_corr = corr_L * corr_Q * corr_C * corr_G * corr_F

    pair_L = corr_L * (corr_Q + corr_C + corr_G + corr_F)
    pair_Q = corr_Q * (corr_L + corr_C + corr_G + corr_F)
    pair_C = corr_C * (corr_L + corr_Q + corr_G + corr_F)
    pair_G = corr_G * (corr_L + corr_Q + corr_C + corr_F)
    pair_F = corr_F * (corr_L + corr_Q + corr_C + corr_G)

    if pair_bonus_player is not None:
        lamL, lamQ, lamC, lamG, lamF = pair_bonus_player
    else:
        lamL = lamQ = lamC = lamG = lamF = float(pair_bonus)

    αL, αQ, αC, αG, αF = acc_reward
    β = coop_bonus

    uL_full = αL * corr_L + lamL * pair_L + β * all_corr
    uQ_full = αQ * corr_Q + lamQ * pair_Q + β * all_corr
    uC_full = αC * corr_C + lamC * pair_C + β * all_corr
    uG_full = αG * corr_G + lamG * pair_G + β * all_corr
    uF_full = αF * corr_F + lamF * pair_F + β * all_corr

    # 精确期望（枚举 2^5=32）
    wQ = piQ.unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5)
    wC = piC.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5)
    wG = piG.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(5)
    wF_base = piF.detach() if detach_F_in_game else piF
    wF = wF_base.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    wL = piL.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

    uL_a = (uL_full * wQ * wC * wG * wF).sum(dim=(2, 3, 4, 5))       # (B,2)
    uQ_a = (uQ_full * wL * wC * wG * wF).sum(dim=(1, 3, 4, 5))       # (B,2)
    uC_a = (uC_full * wL * wQ * wG * wF).sum(dim=(1, 2, 4, 5))       # (B,2)
    uG_a = (uG_full * wL * wQ * wC * wF).sum(dim=(1, 2, 3, 5))       # (B,2)
    uF_a = (uF_full * wL * wQ * wC * wG).sum(dim=(1, 2, 3, 4))       # (B,2)

    def _log(p):
        return torch.log(p.clamp_min(1e-8))

    # F_i = -u_i + τ log π_i
    FiL = -uL_a + tau_reg * _log(piL)
    FiQ = -uQ_a + tau_reg * _log(piQ)
    FiC = -uC_a + tau_reg * _log(piC)
    FiG = -uG_a + tau_reg * _log(piG)
    FiF = -uF_a + tau_reg * _log(piF)

    def _expand_xhat(xhat_i: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if xhat_i.dim() == 2 and xhat_i.size(1) == ref.size(1):
            if xhat_i.size(0) == 1:
                return xhat_i.expand_as(ref).to(ref.device, ref.dtype)
            if xhat_i.size(0) == ref.size(0):
                return xhat_i.to(ref.device, ref.dtype)
        raise ValueError("custom xhat_i must be (1,2) or (B,2)")

    def _get_xhat(pi: torch.Tensor, which: str, custom: Optional[torch.Tensor]) -> torch.Tensor:
        which = which.lower()
        if which == "current":
            return pi
        if which == "uniform":
            return torch.full_like(pi, 0.5)
        if which == "custom":
            if custom is None:
                raise ValueError("xhat_custom is required when xhat_type='custom'")
            return _expand_xhat(custom, pi)
        raise ValueError(f"Unknown xhat_type: {which}")

    xhatL = _get_xhat(piL, xhat_type, None if xhat_custom is None else xhat_custom[0])
    xhatQ = _get_xhat(piQ, xhat_type, None if xhat_custom is None else xhat_custom[1])
    xhatC = _get_xhat(piC, xhat_type, None if xhat_custom is None else xhat_custom[2])
    xhatG = _get_xhat(piG, xhat_type, None if xhat_custom is None else xhat_custom[3])
    xhatF = _get_xhat(piF, xhat_type, None if xhat_custom is None else xhat_custom[4])

    baseL = (FiL * xhatL).sum(dim=-1, keepdim=True)
    baseQ = (FiQ * xhatQ).sum(dim=-1, keepdim=True)
    baseC = (FiC * xhatC).sum(dim=-1, keepdim=True)
    baseG = (FiG * xhatG).sum(dim=-1, keepdim=True)
    baseF = (FiF * xhatF).sum(dim=-1, keepdim=True)

    AL = FiL - baseL
    AQ = FiQ - baseQ
    AC = FiC - baseC
    AG = FiG - baseG
    AF = FiF - baseF

    # 线性 NAL：< sg[A], π >
    loss_L = (AL.detach() * piL).sum(dim=-1).mean()
    loss_Q = (AQ.detach() * piQ).sum(dim=-1).mean()
    loss_C = (AC.detach() * piC).sum(dim=-1).mean()
    loss_G = (AG.detach() * piG).sum(dim=-1).mean()
    loss_F = (AF.detach() * piF).sum(dim=-1).mean()

    denom = float(sum(loss_weight))
    game_loss = (
        loss_weight[0] * loss_L
        + loss_weight[1] * loss_Q
        + loss_weight[2] * loss_C
        + loss_weight[3] * loss_G
        + loss_weight[4] * loss_F
    ) / max(denom, 1e-8)
    game_loss = torch.nan_to_num(game_loss, nan=0.0)

    diag = {
        "A_mean_L": AL.mean().item(),
        "A_mean_Q": AQ.mean().item(),
        "A_mean_C": AC.mean().item(),
        "A_mean_G": AG.mean().item(),
        "A_mean_F": AF.mean().item(),
        "<pi,A>_L": (piL * AL).sum(dim=-1).mean().item(),
        "<pi,A>_Q": (piQ * AQ).sum(dim=-1).mean().item(),
        "<pi,A>_C": (piC * AC).sum(dim=-1).mean().item(),
        "<pi,A>_G": (piG * AG).sum(dim=-1).mean().item(),
        "<pi,A>_F": (piF * AF).sum(dim=-1).mean().item(),
        "u_mean_L": uL_a.mean().item(),
        "u_mean_Q": uQ_a.mean().item(),
        "u_mean_C": uC_a.mean().item(),
        "u_mean_G": uG_a.mean().item(),
        "u_mean_F": uF_a.mean().item(),
        "tau_reg": float(tau_reg),
        "xhat_type": str(xhat_type),
        "pair_bonus_LQCGF": (float(lamL), float(lamQ), float(lamC), float(lamG), float(lamF)),
        "coop_bonus": float(β),
        "detach_F_in_game": bool(detach_F_in_game),
    }
    return game_loss, diag


# -----------------------------
# 训练器（只保留博弈 + KL_ref 正则）
# -----------------------------
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

        self.model = FourBackbones(
            clip_dim=cfg.get("clip_dim", 512),
            proj_dim=cfg["proj_dim"],
            clip_out_dim=cfg["proj_dim"],
            llava_dim=cfg.get("llava_dim", 4096),
            qwen_dim=cfg.get("qwen_dim", 3584),
            gemma_dim=cfg.get("gemma_dim", 2560),
            mlp_dropout=cfg.get("mlp_dropout", 0.1),
            max_img_len=cfg.get("max_img_len", 577),
            max_txt_len=cfg.get("max_txt_len", 77),
        ).to(self.device)

        clip_params = list(self.model.clip_enc.parameters()) + list(self.model.pi_C.parameters())
        other_params = (
            list(self.model.llava_proj.parameters())
            + list(self.model.qwen_proj.parameters())
            + list(self.model.gemma_proj.parameters())
            + list(self.model.pi_L.parameters())
            + list(self.model.pi_Q.parameters())
            + list(self.model.pi_G.parameters())
            + list(self.model.pi_F.parameters())
            + list(self.model.clf.parameters())  # 仅用于 kl_ref="teacher" 时提供 logits（不参与损失）
        )

        self.opt = torch.optim.AdamW(
            [
                {"params": other_params, "lr": cfg["lr"]},
                {"params": clip_params, "lr": cfg.get("lr_clip", cfg["lr"] * 3)},
            ],
            weight_decay=0.01,
        )

        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg["epochs"], eta_min=1e-6)

        self.best_acc = -1.0
        self.best_metrics = {"acc": -1.0, "f1": -1.0, "auc": -1.0}
        self.patience = cfg.get("patience", 10)
        self.wait = 0

        self.freeze_LQG_epochs = cfg.get("freeze_LQG_epochs", 0)

        # ===== KL_ref (Jeffreys-β) 配置 =====
        self.kl_on = bool(self.cfg.get("kl_on", False))
        self.kl_beta = float(self.cfg.get("kl_beta", 0.0))      # KL 损失系数
        self.kl_target = self.cfg.get("kl_target", None)        # 目标 Jβ
        self.kl_adaptive = bool(self.cfg.get("kl_adaptive", False))
        self.kl_ref_mode = str(self.cfg.get("kl_ref", "uniform")).lower()    # uniform | teacher | ema_model
        self.kl_players = set(self.cfg.get("kl_players", ["F"]))
        self.kl_ema_rho = float(self.cfg.get("kl_ema_rho", 0.05))
        self.kl_mix = float(self.cfg.get("kl_mix", 0.5))        # Jeffreys-β 的 β

        self.ref_model = None
        if self.kl_on and self.kl_ref_mode == "ema_model":
            self.ref_model = deepcopy(self.model).to(self.device)
            for p in self.ref_model.parameters():
                p.requires_grad = False

    def _maybe_freeze_unfreeze(self, ep: int):
        if self.freeze_LQG_epochs <= 0:
            return

        if ep == 1:
            for p in self.model.llava_proj.parameters(): p.requires_grad = False
            for p in self.model.qwen_proj.parameters(): p.requires_grad = False
            for p in self.model.gemma_proj.parameters(): p.requires_grad = False
            for p in self.model.pi_L.parameters(): p.requires_grad = False
            for p in self.model.pi_Q.parameters(): p.requires_grad = False
            for p in self.model.pi_G.parameters(): p.requires_grad = False
            # ⚠️ 不建议冻结 pi_F（最终 logits）
            # for p in self.model.pi_F.parameters(): p.requires_grad = False

        if ep == self.freeze_LQG_epochs + 1:
            for p in self.model.llava_proj.parameters(): p.requires_grad = True
            for p in self.model.qwen_proj.parameters(): p.requires_grad = True
            for p in self.model.gemma_proj.parameters(): p.requires_grad = True
            for p in self.model.pi_L.parameters(): p.requires_grad = True
            for p in self.model.pi_Q.parameters(): p.requires_grad = True
            for p in self.model.pi_G.parameters(): p.requires_grad = True
            for p in self.model.pi_F.parameters(): p.requires_grad = True

    @staticmethod
    def _entropy(p: torch.Tensor) -> float:
        p = p.clamp_min(1e-8)
        return float((-(p * p.log()).sum(dim=-1).mean()).item())

    @staticmethod
    def _kl_forward(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """KL(q||p)"""
        q = q.clamp(min=eps, max=1.0)
        p = p.clamp(min=eps, max=1.0)
        res = (q * (torch.log(q) - torch.log(p))).sum(dim=-1).mean()
        return torch.nan_to_num(res, nan=0.0)

    @staticmethod
    def _kl_reverse(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """KL(p||q)"""
        p = p.clamp(min=eps, max=1.0)
        q = q.clamp(min=eps, max=1.0)
        res = (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()
        return torch.nan_to_num(res, nan=0.0)

    @torch.no_grad()
    def _ema_update_ref(self):
        if self.ref_model is None:
            return
        rho = self.kl_ema_rho
        for p_ref, p in zip(self.ref_model.parameters(), self.model.parameters()):
            if p_ref.data.shape == p.data.shape:
                p_ref.data.mul_(1.0 - rho).add_(p.data, alpha=rho)

    def step(self, batch):
        self.model.train()
        for k in batch:
            batch[k] = batch[k].to(self.device)

        out = self.model(batch)
        labels = batch["label"]

        # 策略分布（温度控制尖锐度）
        piL = F.softmax(out["pi_L"] / self.cfg.get("tau_L", 1.0), dim=-1)
        piQ = F.softmax(out["pi_Q"] / self.cfg.get("tau_Q", 1.0), dim=-1)
        piC = F.softmax(out["pi_C"] / self.cfg.get("tau_C", 1.0), dim=-1)
        piG = F.softmax(out["pi_G"] / self.cfg.get("tau_G", 1.0), dim=-1)
        piF = F.softmax(out["pi_F"] / self.cfg.get("tau_F", 1.0), dim=-1)

        # ===========================
        # 可选 xhat：
        #   - "mix":     xhat=(1-eps)*pi + eps*u
        #   - "uniform": xhat=u
        # ===========================
        xhat_mode = str(self.cfg.get("xhat_mode", "mix")).lower()

        if xhat_mode == "mix":
            eps_hat = float(self.cfg.get("xhat_eps", 0.05))
            uni = 1.0 / piF.size(-1)  # 二分类=0.5
            uL = torch.full_like(piL, uni)
            uQ = torch.full_like(piQ, uni)
            uC = torch.full_like(piC, uni)
            uG = torch.full_like(piG, uni)
            uF = torch.full_like(piF, uni)

            xhatL = ((1.0 - eps_hat) * piL + eps_hat * uL).detach()
            xhatQ = ((1.0 - eps_hat) * piQ + eps_hat * uQ).detach()
            xhatC = ((1.0 - eps_hat) * piC + eps_hat * uC).detach()
            xhatG = ((1.0 - eps_hat) * piG + eps_hat * uG).detach()
            xhatF = ((1.0 - eps_hat) * piF + eps_hat * uF).detach()

            game_loss, diag = nash_advantage_linear_nal5(
                piL, piQ, piC, piG, piF, labels,
                acc_reward=self.cfg.get("acc_reward_player5", (1.0, 1.0, 1.0, 1.0, 1.0)),
                coop_bonus=self.cfg.get("coop_bonus", 1.0),
                tau_reg=self.cfg.get("tau_reg", 0.0),
                xhat_type="custom",
                xhat_custom=(xhatL, xhatQ, xhatC, xhatG, xhatF),
                loss_weight=self.cfg.get("loss_weight_player5", (1.0, 1.0, 1.0, 1.0, 1.0)),
                pair_bonus=self.cfg.get("pair_bonus", 0.0),
                pair_bonus_player=self.cfg.get("pair_bonus_player5", None),
                detach_F_in_game=self.cfg.get("detach_F_in_game", False),
            )

        elif xhat_mode == "uniform":
            eps_hat = 0.0
            game_loss, diag = nash_advantage_linear_nal5(
                piL, piQ, piC, piG, piF, labels,
                acc_reward=self.cfg.get("acc_reward_player5", (1.0, 1.0, 1.0, 1.0, 1.0)),
                coop_bonus=self.cfg.get("coop_bonus", 1.0),
                tau_reg=self.cfg.get("tau_reg", 0.0),
                xhat_type="uniform",
                xhat_custom=None,
                loss_weight=self.cfg.get("loss_weight_player5", (1.0, 1.0, 1.0, 1.0, 1.0)),
                pair_bonus=self.cfg.get("pair_bonus", 0.0),
                pair_bonus_player=self.cfg.get("pair_bonus_player5", None),
                detach_F_in_game=self.cfg.get("detach_F_in_game", False),
            )
        else:
            raise ValueError(f"Unknown xhat_mode={xhat_mode}. Use 'mix' or 'uniform'.")

        # ===========================
        # KL_ref: Jeffreys-β on player F
        # ===========================
        kl_val = torch.tensor(0.0, device=self.device)
        kl_loss = torch.tensor(0.0, device=self.device)
        kl_fwd = torch.tensor(0.0, device=self.device)  # KL(q||p)
        kl_rev = torch.tensor(0.0, device=self.device)  # KL(p||q)

        if self.kl_on and ("F" in self.kl_players):
            # 参考分布 qF
            if self.kl_ref_mode == "uniform":
                qF = torch.full_like(piF, 1.0 / piF.size(-1))
            elif self.kl_ref_mode == "teacher":
                # 注意：本脚本删了 CE，因此 clf 未必“学到”任何东西；teacher 分布可能无意义
                qF = F.softmax(out["logits_clf"].detach() / self.cfg.get("tau_F", 1.0), dim=-1)
            elif self.kl_ref_mode == "ema_model":
                if self.ref_model is None:
                    raise RuntimeError("kl_ref='ema_model' requires ref_model, but ref_model is None.")
                with torch.no_grad():
                    ref_out = self.ref_model(batch)
                    qF = F.softmax(ref_out["pi_F"] / self.cfg.get("tau_F", 1.0), dim=-1)
            else:
                raise ValueError(f"Unknown kl_ref: {self.kl_ref_mode}")

            kl_fwd = self._kl_forward(qF, piF)  # KL(q||p)
            kl_rev = self._kl_reverse(piF, qF)  # KL(p||q)
            beta_mix = float(self.kl_mix)
            kl_val = (1.0 - beta_mix) * kl_fwd + beta_mix * kl_rev   # Jβ
            kl_loss = float(self.kl_beta) * kl_val

        ce_loss = F.cross_entropy(out["pi_F"], labels)
        loss = float(self.cfg.get("game_w", 1.0)) * game_loss + kl_loss + ce_loss
        loss = torch.nan_to_num(loss, nan=0.0)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        # EMA 参考模型更新
        if self.kl_on and self.kl_ref_mode == "ema_model":
            self._ema_update_ref()

        # 自适应 KL 系数：把 Jβ 推向 kl_target
        if self.kl_on and self.kl_adaptive and (self.kl_target is not None):
            with torch.no_grad():
                targ = float(self.kl_target)
                if kl_val.item() > 1.2 * targ:
                    self.kl_beta *= 1.15
                elif kl_val.item() < targ / 1.2:
                    self.kl_beta /= 1.15
                self.kl_beta = float(np.clip(self.kl_beta, 1e-6, 10.0))

        with torch.no_grad():
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

            piC_entropy = self._entropy(piC)
            piF_entropy = self._entropy(piF)
            gate_mean = out["clip_dbg"]["gate_mean"].mean().item()
            alpha2 = out["clip_dbg"]["keyless_alpha2"].mean(dim=0)

        return {
            "loss": float(loss.item()),
            "game_loss": float(game_loss.item()),
            "acc": acc,

            "xhat_mode": xhat_mode,
            "xhat_eps": float(eps_hat),

            "kl_F": float(kl_val.item()),         # Jβ
            "kl_F_fwd": float(kl_fwd.item()),     # KL(q||p)
            "kl_F_rev": float(kl_rev.item()),     # KL(p||q)
            "kl_beta": float(self.kl_beta),
            "kl_mix": float(self.kl_mix),
            "kl_ref": self.kl_ref_mode,

            **diag,
            "gate_mean": gate_mean,
            "alpha2_q0": alpha2[0].item(),
            "alpha2_q1": alpha2[1].item(),
            "piC_entropy": piC_entropy,
            "piF_entropy": piF_entropy,
        }

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_y, all_pred, all_prob = [], [], []
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            out = self.model(batch)
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            all_y.append(batch["label"].cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs[:, 1].cpu().numpy())

        y = np.concatenate(all_y)
        yhat = np.concatenate(all_pred)
        phat = np.concatenate(all_prob)

        acc = accuracy_score(y, yhat)
        f1 = f1_score(y, yhat, average="macro")
        try:
            auc = roc_auc_score(y, phat) if len(np.unique(y)) > 1 else 0.5
        except Exception:
            auc = 0.5
        return {"acc": acc, "f1": f1, "auc": auc}

    def train(self, train_loader, valid_loader=None):
        print(f"Device: {self.device}")
        for ep in range(1, self.cfg["epochs"] + 1):
            self._maybe_freeze_unfreeze(ep)

            pbar = tqdm(train_loader, desc=f"Epoch {ep}/{self.cfg['epochs']}")
            logs = {"loss": [], "game": [], "acc": [], "kl": []}

            for batch in pbar:
                m = self.step(batch)
                logs["loss"].append(m["loss"])
                logs["game"].append(m["game_loss"])
                logs["acc"].append(m["acc"])
                logs["kl"].append(m["kl_F"])

                pbar.set_postfix({
                    "loss": f"{np.mean(logs['loss']):.3f}",
                    "acc": f"{np.mean(logs['acc']):.3f}",
                    "game": f"{np.mean(logs['game']):.3f}",
                    "Jβ": f"{np.mean(logs['kl']):.4f}",
                    "xhat": f"{m['xhat_mode']}:{m['xhat_eps']:.3f}",
                    "ref": f"{m['kl_ref']}",
                    "kβ": f"{m['kl_beta']:.3g}",
                })

            self.sched.step()

            if valid_loader is not None:
                val = self.evaluate(valid_loader)
                best_str = "N/A/N/A/N/A" if self.best_acc < 0 else \
                           f"{self.best_metrics['acc']:.4f}/{self.best_metrics['f1']:.4f}/{self.best_metrics['auc']:.4f}"

                print(
                    f"[Epoch {ep}] "
                    f"train_loss={np.mean(logs['loss']):.4f} train_acc={np.mean(logs['acc']):.4f} "
                    f"| VAL acc={val['acc']:.4f} f1={val['f1']:.4f} auc={val['auc']:.4f} "
                    f"| BEST(acc/f1/auc)={best_str} "
                    f"| xhat={self.cfg.get('xhat_mode','mix')} eps={self.cfg.get('xhat_eps',0.05):.3f} "
                    f"| KL_ref={self.kl_ref_mode} mix={self.kl_mix:.2f} coef={self.kl_beta:.4g}"
                )

                if val["acc"] > self.best_acc:
                    self.best_metrics = {"acc": val["acc"], "f1": val["f1"], "auc": val["auc"]}
                    self.best_acc = val["acc"]
                    self.wait = 0
                    self.save(self.cfg["ckpt"])
                    print(f"  ✅ New best ACC={self.best_acc:.4f} (checkpoint saved)")
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print(f"  ⏹ Early stop at epoch {ep}.")
                        break

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "cfg": self.cfg}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])


# -----------------------------
# 主流程
# -----------------------------
def main(cfg):
    print("🚀 Loading dataset ...")
    clip_dim = cfg.get("clip_dim", 512)
    train_set = FourPlayerDataset(cfg["train_json"], cfg["data_dir"], max_seq_img=cfg["max_img_len"], max_seq_txt=cfg["max_txt_len"], clip_dim=clip_dim, preload=True)
    test_set = FourPlayerDataset(cfg["test_json"], cfg["data_dir"], max_seq_img=cfg["max_img_len"], max_seq_txt=cfg["max_txt_len"], clip_dim=clip_dim, preload=True)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(cfg)
    trainer.train(train_loader, test_loader)

    print(f"\n📦 Loading best checkpoint from: {cfg['ckpt']}")
    trainer.load(cfg["ckpt"])

    print("\n🎯 Final Evaluation on Test:")
    final = trainer.evaluate(test_loader)
    for k, v in final.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    raise SystemExit("Please edit cfg in run_geco.py and run: python run_geco.py")
