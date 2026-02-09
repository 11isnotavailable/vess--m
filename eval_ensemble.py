import torch
import torch.nn.functional as F
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from src.engine.pl_module import VesselSystem
from tqdm import tqdm

@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    # 1. 定义所有要集成的权重路径
    
    # ckpt_root = Path("/root/autodl-tmp/tb_logs/vessel_experiment_v2/version_14/checkpoints/")
    # ckpt_files = [
    #     "epoch=epoch=150-step=step=2114.ckpt",
    #     "epoch=epoch=256-step=step=3598.ckpt",
    #     "epoch=epoch=370-step=step=5194.ckpt",
    #     "last.ckpt"
    # ]
    # ckpt_root = Path("/root/autodl-tmp/tb_logs/vessel_experiment_v2/version_15/checkpoints/")
    # ckpt_files = [
    #     "epoch=epoch=154-step=step=2170.ckpt",
    #     "epoch=epoch=50-step=step=714.ckpt",
    #     "epoch=epoch=566-step=step=7938.ckpt",
    #     "last.ckpt"
    # ]
    ckpt_root = Path("/root/autodl-tmp/tb_logs/vessel_experiment_v2/version_16/checkpoints/")
    ckpt_files = [
        "epoch=epoch=208-step=step=2926.ckpt",
        "epoch=epoch=225-step=step=3164.ckpt",
        "epoch=epoch=95-step=step=1344.ckpt",
        "last.ckpt"
    ]
    ckpt_paths = [ckpt_root / f for f in ckpt_files]

    # 2. 依次加载模型并存入列表
    models = []
    print(f"📦 正在加载 {len(ckpt_paths)} 个集成权重...")
    for path in ckpt_paths:
        if not path.exists():
            print(f"⚠️ 警告: 找不到权重 {path}")
            continue
        base_model = hydra.utils.instantiate(cfg.model)
        system = VesselSystem.load_from_checkpoint(
            path, model=base_model, cfg=cfg, map_location="cuda:0", strict=False
        )
        system.eval().cuda()
        models.append(system)

    # 3. 数据准备
    OmegaConf.set_struct(cfg, False)
    cfg.data.path = to_absolute_path(cfg.data.path)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    all_metrics = []
    print(f"📈 正在执行软投票集成评估 (样本数: {len(test_loader)})...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            img, mask, _ = batch
            img = img.cuda()
            
            # 存储每个模型的概率图输出
            ensemble_probs = []
            
            for system in models:
                # 滑动窗口推理
                def model_forward(x):
                    out = system.model(x)
                    return out[0] if isinstance(out, (tuple, list)) else out
                
                # 获取该模型的 Logits 并转为概率 (Sigmoid)
                patch_logits = system.inferer(img, model_forward)
                ensemble_probs.append(patch_logits.sigmoid())
            
            # 核心：软投票（对所有模型的概率图取算术平均）
            avg_prob = torch.stack(ensemble_probs).mean(dim=0)
            
            # 使用平均后的概率计算全套指标
            # calculate_all 内部会处理 > 0.5 的二值化逻辑
            metrics = models[0].evaluator.calculate_all(avg_prob, mask)
            all_metrics.append(metrics)

    # 4. 计算并打印集成后的平均值
    print("\n" + "="*40)
    print("🔥 软投票集成 (Ensemble) 最终指标汇总")
    print("="*40)
    
    keys = ["Dice", "Precision", "Recall", "HD95", "clDice"]
    avg_results = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
    
    for k in keys:
        print(f"{k:15s}: {avg_results[k]:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()