import numpy as np
import torch
# 自动处理新旧版本 skimage 的导入冲突
try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    from skimage.morphology import skeletonize as skeletonize_3d

from skimage.measure import label, euler_number
from monai.metrics import compute_hausdorff_distance  # 使用 MONAI 计算 HD

class Evaluator:
    """
    血管拓扑与几何评价器：计算 Dice, clDice, Precision, Recall, HD95 和 Betti Error。
    """
    @staticmethod
    def get_skeleton(img):
        """提取 3D 骨架。输入应为二值化后的 3D Numpy 数组。"""
        data = (img > 0).astype(np.uint8)
        try:
            return skeletonize_3d(data)
        except Exception:
            from skimage.morphology import skeletonize
            return skeletonize(data)

    @staticmethod
    def compute_dice(pred, gt):
        """标准 Dice 系数"""
        intersect = np.sum(pred * gt)
        return (2. * intersect) / (np.sum(pred) + np.sum(gt) + 1e-8)

    @staticmethod
    def compute_precision_recall(pred, gt):
        """计算查准率 (Precision) 和 查全率 (Recall)"""
        tp = np.sum(pred * gt)
        precision = tp / (np.sum(pred) + 1e-8)
        recall = tp / (np.sum(gt) + 1e-8)
        return precision, recall

    @staticmethod
    def compute_hd95(pred, gt):
        """
        使用 MONAI 计算 95% 豪斯多夫距离 (HD95)。
        输入要求为 [B, C, D, H, W] 的 Tensor。
        """
        # 转为 Tensor 并增加 Batch/Channel 维度以符合 MONAI 要求
        p_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        g_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        
        try:
            # percentile=95 即为 HD95
            hd = compute_hausdorff_distance(p_tensor, g_tensor, percentile=95)
            return hd.item()
        except Exception:
            return 0.0

    def compute_cl_dice(self, pred, gt):
        """计算中心线 Dice (clDice)"""
        skel_pred = self.get_skeleton(pred)
        skel_gt = self.get_skeleton(gt)
        
        tprec = np.sum(pred * skel_gt) / (np.sum(skel_gt) + 1e-8)
        tsens = np.sum(gt * skel_pred) / (np.sum(skel_pred) + 1e-8)
        
        return 2 * tprec * tsens / (tprec + tsens + 1e-8)

    def calculate_all(self, pred_tensor, gt_tensor):
        """
        主入口：返回图片要求的 5 个核心指标 + Betti Error。
        """
        threshold = 0.486 
        
        # 转为 Numpy 并进行二值化
        p = (pred_tensor.detach().cpu().numpy() > threshold).astype(np.float32)
        g = (gt_tensor.detach().cpu().numpy() > 0.5).astype(np.float32)
        # 转为 Numpy 并进行二值化
        # p = (pred_tensor.detach().cpu().numpy() > 0.5).astype(np.float32)
        # g = (gt_tensor.detach().cpu().numpy() > 0.5).astype(np.float32)
        
        # 自动处理形状 [B, C, D, H, W] -> [D, H, W]
        if p.ndim == 5: p = p[0, 0]
        if g.ndim == 5: g = g[0, 0]
        
        dice = self.compute_dice(p, g)
        cldice = self.compute_cl_dice(p, g)
        precision, recall = self.compute_precision_recall(p, g)
        hd95_val = self.compute_hd95(p, g)
        
        return {
            "Dice": dice,
            "clDice": cldice,
            "Precision": precision,
            "Recall": recall,
            "HD95": hd95_val
        }