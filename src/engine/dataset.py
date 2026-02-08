# src/engine/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
import lightning.pytorch as pl
from monai.transforms import Compose
import hydra  # 使用 hydra 的实例化工具

class H5Dataset(Dataset):
    """
    通用 H5 数据集类，适配 .nii.h5 及 _withlabel 等复杂命名逻辑。
    """
    def __init__(self, root_dir, list_file, transforms, image_key="image", label_key="label"):
        self.root_dir = Path(root_dir)
        self.image_key = image_key
        self.label_key = label_key
        self.transforms = transforms
        
        # 加载样本清单
        list_path = self.root_dir / list_file
        if not list_path.exists():
            raise FileNotFoundError(f"清单文件未找到: {list_path}")
            
        with open(list_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # 自动处理后缀逻辑
        if not (sample_name.endswith(".h5") or sample_name.endswith(".hdf5")):
            file_path = self.root_dir / f"{sample_name}.h5"
        else:
            file_path = self.root_dir / sample_name

        try:
            with h5py.File(file_path, "r") as f:
                img = f[self.image_key][:].astype(np.float32)
                mask = f[self.label_key][:].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"无法读取文件 {file_path}: {e}")

        # 统一维度为 [C, D, H, W]
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        if mask.ndim == 3:
            mask = mask[np.newaxis, ...]

        # 应用 Transform
        if self.transforms:
            data = self.transforms({"image": img, "label": mask})
            return data["image"], data["label"], sample_name
        
        return torch.from_numpy(img), torch.from_numpy(mask), sample_name

class VesselDataModule(pl.LightningDataModule):
    """
    基于 Lightning 的数据模块，通过 Hydra 递归实例化处理变换。
    """
    def __init__(self, path, val_list="val.txt", test_list="test.txt", 
                 batch_size=2, num_workers=4, transforms=None, **kwargs):
        super().__init__()
        self.path = path
        self.val_list = val_list
        self.test_list = test_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # ✅ 核心修复：直接对 ListConfig 进行实例化
        # Hydra 会自动遍历列表并根据 _target_ 实例化每一个变换，返回对象列表给 Compose
        if transforms:
            if 'train' in transforms:
                # 这种写法不需要循环，Hydra 会处理一切
                self.train_transforms = Compose(hydra.utils.instantiate(transforms.train))
            else:
                self.train_transforms = None
                
            if 'val' in transforms:
                self.val_transforms = Compose(hydra.utils.instantiate(transforms.val))
            else:
                self.val_transforms = None
        else:
            self.train_transforms = None
            self.val_transforms = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = H5Dataset(self.path, "train.txt", self.train_transforms)
            self.val_ds = H5Dataset(self.path, self.val_list, self.val_transforms)
        
        if stage == "test":
            self.test_ds = H5Dataset(self.path, self.test_list, self.val_transforms)

    def train_dataloader(self):
        # 4090 性能强劲，建议将 num_workers 设为 8 或 12
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=16,      # ✅ 增加此参数
            pin_memory=True,    # ✅ 开启内存锁页，加快 Tensor 传输到 GPU
            persistent_workers=True # ✅ 保持进程，避免每个 Epoch 重新启动
        )
    
    def val_dataloader(self):
        # 验证集也可以适当增加
        return DataLoader(
            self.val_ds, 
            batch_size=1, 
            num_workers=8,      # ✅ 验证集也可以加一些
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=1)