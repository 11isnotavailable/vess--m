import os
import torch
import nibabel as nib
import numpy as np
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from src.engine.pl_module import VesselSystem

@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    # 1. 路径与模型加载
    ckpt_path = Path("/root/autodl-tmp/tb_logs/vessel_experiment_v2/version_8/checkpoints/epoch=571-step=8000.ckpt")
    model = hydra.utils.instantiate(cfg.model)
    system = VesselSystem.load_from_checkpoint(ckpt_path, model=model, cfg=cfg, map_location="cuda:0", strict=False)
    system.eval().cuda()

    # 2. 修正路径配置
    OmegaConf.set_struct(cfg, False)
    cfg.data.path = to_absolute_path(cfg.data.path)
    
    # 3. 实例化数据
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    output_dir = Path(to_absolute_path("outputs/inference_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 正在处理 {len(test_loader)} 个样本...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img, mask, name = batch
            
            # 包装前向传播
            def model_forward(x):
                out = system.model(x)
                return out[0] if isinstance(out, (tuple, list)) else out

            # 执行滑动窗口推理
            pred_mask = system.inferer(img.cuda(), model_forward)
            pred_prob = pred_mask.sigmoid().cpu().numpy()[0, 0]
            
            # 保存
            clean_name = os.path.basename(name[0]).split('.')[0]
            save_path = output_dir / f"{clean_name}_pred.nii.gz"
            nib.save(nib.Nifti1Image((pred_prob > 0.5).astype(np.uint8), np.eye(4)), save_path)
            print(f"✅ 已保存: {save_path}")

if __name__ == "__main__":
    main()