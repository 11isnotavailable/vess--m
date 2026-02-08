import numpy as np
import torch
# 自动处理新旧版本 skimage 的导入冲突
try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    # 新版 skimage (0.19+) 将 3D 骨架化整合进了 skeletonize
    from skimage.morphology import skeletonize as skeletonize_3d

from skimage.measure import label, euler_number

class Evaluator:
    """
    血管拓扑评价器：计算 Dice, clDice 和 Betti Error。
    针对 3D 血管数据进行了性能优化。
    """
    @staticmethod
    def get_skeleton(img):
        """
        提取 3D 骨架。输入应为二值化后的 3D Numpy 数组。
        """
        # 确保输入是 bool 或 uint8，这对算法效率至关重要
        data = (img > 0).astype(np.uint8)
        try:
            # 优先尝试 Lee 算法 (传统 3D 骨架化)
            return skeletonize_3d(data)
        except Exception:
            # 万能兜底：调用最新的统一接口
            from skimage.morphology import skeletonize
            return skeletonize(data)

    @staticmethod
    def compute_dice(pred, gt):
        """标准 Dice 系数"""
        intersect = np.sum(pred * gt)
        return (2. * intersect) / (np.sum(pred) + np.sum(gt) + 1e-8)

    def compute_cl_dice(self, pred, gt):
        """
        计算中心线 Dice (clDice)。
        公式：$clDice = 2 \cdot \frac{TPrec \cdot TSens}{TPrec + TSens}$
        """
        skel_pred = self.get_skeleton(pred)
        skel_gt = self.get_skeleton(gt)
        
        # TPrec: 预测结果覆盖真实骨架的比例
        tprec = np.sum(pred * skel_gt) / (np.sum(skel_gt) + 1e-8)
        # TSens: 真实结果覆盖预测骨架的比例
        tsens = np.sum(gt * skel_pred) / (np.sum(skel_pred) + 1e-8)
        
        return 2 * tprec * tsens / (tprec + tsens + 1e-8)

    def compute_betti_error(self, pred, gt):
        """
        计算贝蒂数误差 (Betti Error)。
        b0: 连通分量数量 (衡量血管是否断裂)
        b1: 环路数量 (衡量血管是否误连)
        """
        def get_betti(img):
            # 使用 26 连通域 (connectivity=3) 处理 3D 体素
            _, b0 = label(img, return_num=True, connectivity=3)
            ec = euler_number(img, connectivity=3)
            # 拓扑公式：Euler Characteristic = b0 - b1 + b2 (b2 在薄壁血管中通常忽略)
            return b0, b0 - ec
        
        p_b0, p_b1 = get_betti(pred)
        g_b0, g_b1 = get_betti(gt)
        return abs(p_b0 - g_b0), abs(p_b1 - g_b1)

    def calculate_all(self, pred_tensor, gt_tensor):
        """
        主入口：输入模型输出的概率 Tensor，返回全套指标。
        """
        # 转为 Numpy 并进行二值化
        p = (pred_tensor.detach().cpu().numpy() > 0.5).astype(np.float32)
        g = (gt_tensor.detach().cpu().numpy() > 0.5).astype(np.float32)
        
        # 自动处理 PyTorch 的 5D 形状 [B, C, D, H, W] -> [D, H, W]
        if p.ndim == 5: p = p[0, 0]
        if g.ndim == 5: g = g[0, 0]
        
        dice = self.compute_dice(p, g)
        cldice = self.compute_cl_dice(p, g)
        b0_err, b1_err = self.compute_betti_error(p, g)
        
        return {
            "dice": dice,
            "cldice": cldice,
            "betti0_error": b0_err,
            "betti1_error": b1_err
        }