import numpy as np
from skimage.morphology import skeletonize_3d
from sklearn.metrics import confusion_matrix

class Evaluator:
    """
    计算 Dice, clDice 以及 3D 血管连通性指标。
    """
    @staticmethod
    def cl_dice(v_p, v_l):
        """
        计算拓扑感知中心线 Dice [cite: 409]。
        """
        def cl_score(v, s):
            return np.sum(v * s) / (np.sum(s) + 1e-8)

        v_p = (v_p > 0.5).astype(np.uint8)
        v_l = (v_l > 0.5).astype(np.uint8)
        
        # 提取 3D 骨架
        skel_p = skeletonize_3d(v_p)
        skel_l = skeletonize_3d(v_l)
        
        tprec = cl_score(v_p, skel_l)
        tsens = cl_score(v_l, skel_p)
        return 2 * tprec * tsens / (tprec + tsens + 1e-8)

    def estimate_metrics(self, pred, gt):
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        # 基础 Dice 计算
        intersect = np.sum(pred_np * gt_np)
        dice = (2 * intersect) / (np.sum(pred_np) + np.sum(gt_np) + 1e-8)
        
        # 计算 clDice
        cldice = self.cl_dice(pred_np, gt_np)
        
        return {"dice": dice, "cldice": cldice}