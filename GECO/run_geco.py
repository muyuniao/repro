from train import main


cfg = {
    "train_json": "./train_PrideMM_with_hidden5.json",
    "test_json": "./test_PrideMM1_with_hidden5.json",
    "data_dir": "./hidden_states_Pridemm/",
    "batch": 64,
    "epochs": 5000,
    "lr": 2e-5,
    "lr_clip": 2e-5,
    "device": "cuda",
    "proj_dim": 768,
    "patience": 2000,
    "game_w": 1.0,
    "tau_L": 1.0, "tau_Q": 1.0, "tau_C": 1.0, "tau_G": 1.0, "tau_F": 1.0,
    "acc_reward_player5": (1.0, 1.0, 1.0, 1.0, 1.0),
    "loss_weight_player5": (1.0, 1.0, 1.0, 1.0, 1.0),
    "pair_bonus": 1.0,
    "coop_bonus": 1.0,
    "tau_reg": 0.25,
    "xhat_mode": "mix",   # or "uniform"
    "xhat_eps": 0.05,     # only for "mix"
    "detach_F_in_game": False,
    "llava_dim": 4096,
    "qwen_dim": 3584,
    "gemma_dim": 2560,
    "ckpt": "./ckpts/five_player_nal_best.pth",
    "max_img_len": 577,
    "max_txt_len": 77,
    "num_workers": 4,
    "freeze_LQG_epochs": 0,
    "mlp_dropout": 0.5,
    "kl_on": True,
    "kl_players": ["F"],         # 先只对 F
    "kl_beta": 1.0,              # 初始系数
    "kl_adaptive": True,
    "kl_target": 0.001,           # 目标 Jβ
    "kl_mix": 0.5,               # β: 1=纯反向KL(p||q), 0=纯前向KL(q||p)
    # uniform | teacher | ema_model
    "kl_ref": "ema_model",
    "kl_ema_rho": 0.05,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pair_bonus", type=float, default=1.0)
    parser.add_argument("--coop_bonus", type=float, default=1.0)
    parser.add_argument("--kl_target", type=float, default=0.001)
    parser.add_argument("--ckpt", type=str, default="./ckpts/five_player_nal_best.pth")
    args = parser.parse_args()
    
    cfg["device"] = args.device
    cfg["pair_bonus"] = args.pair_bonus
    cfg["coop_bonus"] = args.coop_bonus
    cfg["kl_target"] = args.kl_target
    cfg["ckpt"] = args.ckpt
    
    main(cfg)
