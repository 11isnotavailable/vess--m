# test_config.py
import hydra
from omegaconf import DictConfig
import torch

@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def test_setup(cfg: DictConfig):
    print("--- Testing Config Loading ---")
    try:
        # 模拟模型参数
        mock_params = [torch.nn.Parameter(torch.randn(2, 2))]
        
        # 1. 测试优化器实例化
        opt_cfg = cfg.trainer.lightning_module.optimizer_factory
        optimizer = hydra.utils.instantiate(opt_cfg, params=mock_params)
        print(f"✅ Optimizer OK: {type(optimizer)}")
        
        # 2. 测试调度器链
        if "scheduler_configs" in cfg.trainer.lightning_module:
            for name, s_cfg in cfg.trainer.lightning_module.scheduler_configs.items():
                scheduler = hydra.utils.instantiate(s_cfg.scheduler, optimizer=optimizer)
                print(f"✅ Scheduler '{name}' OK: {type(scheduler)}")
        
        print("\n--- All Configs Valid! You can run train_v2.py now. ---")
    except Exception as e:
        print(f"❌ Error detected: {e}")

if __name__ == "__main__":
    test_setup()