import os
import torch
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    # 1. 针对 4090 的算力优化
    # 显著提升 3D 卷积在 Tensor Cores 上的运行速度
    torch.set_float32_matmul_precision('high')
    
    # 2. 设置随机种子
    L.seed_everything(cfg.get("seed", 1337))

    # 3. 实例化 DataModule
    # 确保路径被 Hydra 转换为绝对路径，避免路径漂移
    OmegaConf.set_struct(cfg, False)
    from hydra.utils import to_absolute_path
    cfg.data.path = to_absolute_path(cfg.data.path)
    
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")

    # 4. 实例化模型系统
    # 这里通过 hydra 传入 model 配置（baseline 或 fusion）
    model = hydra.utils.instantiate(cfg.model)
    from src.engine.pl_module import VesselSystem
    system = VesselSystem(model=model, cfg=cfg)

    # 5. 配置日志与回调
    logger = TensorBoardLogger(save_dir="tb_logs", name=cfg.run_name)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"tb_logs/{cfg.run_name}/version_{logger.version}/checkpoints",
        filename="epoch={epoch}-step={step}",
        monitor="val/dice",
        mode="max",
        save_last=True,
        save_top_k=3
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 6. 初始化 Trainer
    # 保持 8000 Steps 以确保 36.3M 参数的 Fusion 模型能充分收敛
    trainer = L.Trainer(
        max_steps=cfg.trainer.get("max_steps", 8000), 
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed",  # 使用混合精度节省 4090 显存
        val_check_interval=cfg.trainer.get("val_check_interval", 1.0),
        log_every_n_steps=10
    )

# 1. 手动触发数据准备
    datamodule.setup(stage="fit")

    # 2. 打印确认，确保数据没问题
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    print(f"📦 [Data Check] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. 🚀 终极修复：不再传递 datamodule 实例，直接传递 loaders 关键字参数
    trainer.fit(
        model=system,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    main()