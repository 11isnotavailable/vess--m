import torch
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from src.engine.pl_module import VesselSystem
from tqdm import tqdm

@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    # 1. 加载模型 (修正后的路径)
    ckpt_path = Path("/root/autodl-tmp/tb_logs/vessel_experiment_v2/version_16/checkpoints/epoch=epoch=208-step=step=2926.ckpt")
    #ckpt_path = Path("/root/autodl-tmp/tb_logs/vessel_experiment_v2/version_15/checkpoints/epoch=epoch=154-step=step=2170.ckpt")
    #ckpt_path = Path("/root/autodl-tmp/tb_logs/vessel_experiment_v2/version_14/checkpoints/epoch=epoch=150-step=step=2114.ckpt")
    model = hydra.utils.instantiate(cfg.model)
    system = VesselSystem.load_from_checkpoint(ckpt_path, model=model, cfg=cfg, map_location="cuda:0", strict=False)
    system.eval().cuda()

    # 2. 配置修正
    OmegaConf.set_struct(cfg, False)
    cfg.data.path = to_absolute_path(cfg.data.path)
    
    # 3. 数据准备
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    all_metrics = []
    print(f"📈 正在评估测试集 (共 {len(test_loader)} 个样本)...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            img, mask, _ = batch
            
            def model_forward(x):
                out = system.model(x)
                return out[0] if isinstance(out, (tuple, list)) else out
            
            pred_mask = system.inferer(img.cuda(), model_forward)
            
            # 计算指标
            metrics = system.evaluator.calculate_all(pred_mask.sigmoid(), mask)
            all_metrics.append(metrics)

    # 4. 计算并打印平均值 (对齐图片格式)
    print("\n" + "="*35)
    print("🔥 测试集最终指标汇总")
    print("="*35)
    
    # 获取所有键名
    keys = all_metrics[0].keys()
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
    
    # 按照你要求的顺序打印 (如果键存在的话)
    target_order = ["Dice", "Precision", "Recall", "HD95", "clDice"]
    for k in target_order:
        if k in avg_metrics:
            print(f"{k:15s}: {avg_metrics[k]:.4f}")
    
    print("="*35)

if __name__ == "__main__":
    main()