import torch
import lightning.pytorch as pl
import hydra
from monai.inferers import SlidingWindowInferer
from src.utils.metrics import Evaluator
import torch.nn.functional as F # 引入 F 用于可能的对齐

class VesselSystem(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg = cfg
        
        # 实例化 Loss
        self.main_criterion = hydra.utils.instantiate(cfg.trainer.lightning_module.loss)
        self.aux_criterion = torch.nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator()

        # ✅ 完美重现原版实现：直接从配置读取滑动窗口参数
        lm_cfg = cfg.trainer.lightning_module
        self.inferer = SlidingWindowInferer(
            roi_size=lm_cfg.roi_size,          # [96, 96, 96]
            sw_batch_size=lm_cfg.sw_batch_size, # 4
            overlap=lm_cfg.overlap,            # 0.5
            mode=lm_cfg.mode                   # "gaussian"
        )

        self.prediction_threshold = lm_cfg.prediction_threshold

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask, _ = batch
        pred_mask, aux_dict = self.model(img)
        
        # ✅ 完美对齐：使用 PyTorch 原生切片实现中心裁剪
        # 彻底解决 "ImportError: cannot import name 'center_spatial_crop'"
        if pred_mask.shape != mask.shape:
            # 获取两者的维度差异
            # 假设形状为 [B, C, D, H, W]
            diff_d = (pred_mask.shape[2] - mask.shape[2]) // 2
            diff_h = (pred_mask.shape[3] - mask.shape[3]) // 2
            diff_w = (pred_mask.shape[4] - mask.shape[4]) // 2
            
            # 执行切片裁剪
            pred_mask = pred_mask[:, :, 
                                  diff_d : diff_d + mask.shape[2],
                                  diff_h : diff_h + mask.shape[3],
                                  diff_w : diff_w + mask.shape[4]]
        
        # 计算主分割损失 (Dice + CE)
        loss_seg = self.main_criterion(pred_mask, mask.float())
        
        # 计算拓扑感知辅助损失
        loss_topo = 0
        topo_count = 0
        for aux in aux_dict.values():
            if isinstance(aux, dict) and "P_pred" in aux and "P_pseudo" in aux:
                loss_topo += self.aux_criterion(aux["P_pred"], aux["P_pseudo"])
                topo_count += 1
        
        # 组合总损失
        lambda_topo = getattr(self.cfg.model, "lambda_topo", 0.3)
        total_loss = loss_seg + (lambda_topo * loss_topo if topo_count > 0 else 0)
        
        # 记录日志
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        img, mask, name = batch
        
        # 滑动窗口推理
        pred_mask = self.inferer(img, lambda x: self.model(x)[0])
        
        # 计算 Loss
        loss = self.main_criterion(pred_mask, mask.float())
        
        # 计算指标
        metrics = self.evaluator.calculate_all(pred_mask.sigmoid(), mask)
        
        # ✅ 修改：添加 prog_bar=True 即可在终端进度条实时看到数值
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dice", metrics["Dice"], prog_bar=True)
        self.log("val/cldice", metrics["clDice"], prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        # 1. 实例化优化器配置
        opt_cfg = self.cfg.trainer.lightning_module.optimizer_factory
        optimizer_obj = hydra.utils.instantiate(opt_cfg, params=self.parameters())
        
        # ✅ 核心修复：如果返回的是 partial 对象，执行它以获得真实的优化器
        if isinstance(optimizer_obj, torch.optim.Optimizer):
            optimizer = optimizer_obj
        else:
            # 这是一个 partial 对象，手动调用它
            optimizer = optimizer_obj() 
        
        print(f"[DEBUG] Real Optimizer: {type(optimizer).__name__}")

        # 2. 处理调度器
        if "scheduler_configs" in self.cfg.trainer.lightning_module:
            lr_schedulers = []
            sched_configs = self.cfg.trainer.lightning_module.scheduler_configs
            
            for name, s_cfg in sched_configs.items():
                scheduler_obj = hydra.utils.instantiate(s_cfg.scheduler, optimizer=optimizer)
                
                # ✅ 同样的修复：处理可能存在的 partial 调度器
                if hasattr(scheduler_obj, "__call__") and not isinstance(scheduler_obj, torch.optim.lr_scheduler.LRScheduler):
                    scheduler = scheduler_obj()
                else:
                    scheduler = scheduler_obj

                lr_schedulers.append({
                    "scheduler": scheduler,
                    "interval": s_cfg.interval,
                    "frequency": s_cfg.frequency,
                    "name": f"lr/{name}"
                })
                print(f"[DEBUG] Real Scheduler added: {name}")

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_schedulers
            }
            
        return optimizer