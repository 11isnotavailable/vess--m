import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from src.engine.pl_module import VesselSystem
from tqdm import tqdm

@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    # 1. 加载 8000 步的最优模型
    ckpt_path = Path("/root/autodl-tmp/tb_logs/vessel_experiment_v2/version_9/checkpoints/epoch=epoch=60-step=step=854.ckpt")
    model = hydra.utils.instantiate(cfg.model)
    system = VesselSystem.load_from_checkpoint(ckpt_path, model=model, cfg=cfg, map_location="cuda:0", strict=False)
    system.eval().cuda()

    # 2. 修正数据路径与后缀
    OmegaConf.set_struct(cfg, False)
    cfg.data.path = to_absolute_path(cfg.data.path)
    cfg.data.file_format = "nii.h5" 
    
    # 3. 准备数据加载器
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # 4. 指标统计容器
    all_metrics = []
    print(f"📈 正在评估测试集 (共 {len(test_loader)} 个样本)...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            img, mask, _ = batch
            
            # 滑动窗口推理
            def model_forward(x):
                out = system.model(x)
                return out[0] if isinstance(out, (tuple, list)) else out
            
            pred_mask = system.inferer(img.cuda(), model_forward)
            
            # 调用 pl_module 内部集成的 evaluator 计算全套指标
            metrics = system.evaluator.calculate_all(pred_mask.sigmoid(), mask)
            all_metrics.append(metrics)

    # 5. 计算并打印平均值
    print("\n" + "="*30)
    print("🔥 测试集最终指标汇总")
    print("="*30)
    
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    for metric, value in avg_metrics.items():
        print(f"👉 {metric:15s}: {value:.4f}")
    print("="*30)

if __name__ == "__main__":
    import numpy as np
    main()