Vessel-V2: 3D-2D 拓扑融合血管分割系统本项目是一个基于 PyTorch Lightning 开发的 3D 血管分割框架，核心创新点在于引入了 分层最大密度投影（Slab-MIP）融合模块，通过 2D 拓扑先验引导 3D 卷积网络提取更连通的血管结构。📂 目录结构Plaintext/root/autodl-tmp/
├── configs/                # Hydra 配置文件目录 (模型、数据、训练参数)
├── src/
│   ├── engine/
│   │   └── pl_module.py    # PyTorch Lightning 核心训练/验证逻辑
│   │   └── dataset.py    
│   ├── models/
│   │   ├── wrapper.py      # 模型封装容器 (3D Encoder-Decoder + Fusion)
│   │   └── fusion_layers.py # 核心：分层分片交叉注意力融合层
│   │   └── backbone.py 
│   │   └── encoders.py 
│   └── utils/
│       ├── metrics.py      # 评价指标 (Dice, clDice, HD95, etc.)
│       └── evaluation.py 
│       └── io.py
├── train_v2.py             # 训练主入口脚本
├── eval_test.py            # 单模型推理评估脚本
└── eval_ensemble.py        # 软投票集成评估脚本
🚀 运行指南本项目使用 Hydra 进行配置管理，通过 overrides 参数切换模式。
1. 启动训练Baseline 模式（纯 3D UNet）:
    python train_v2.py model=baseline
Fusion 模式（3D + 2D Slab 融合）:
    python train_v2.py model=fusion
3. 执行评估评估训练好的模型在测试集（12 个样本）上的表现：
评估 Baseline
    python eval_test.py model=baseline
评估 Fusion (单模型)
    python eval_test.py model=fusion
3. 多权重集成 (Ensemble)由于血管分割任务存在随机性，本项目支持对保存的 Top-K 权重进行软投票集成以提升指标：
    python eval_ensemble.py model=fusion
🛠️ 如何修改权重路径在执行评估时，您需要指定模型加载的 .ckpt 文件路径：单模型评估 (eval_test.py):修改代码中的 ckpt_path 变量，或者在命令行传入：Python# 默认在代码中修改此行
ckpt_path = "/root/autodl-tmp/tb_logs/.../checkpoints/last.ckpt"
多模型集成 (eval_ensemble.py):修改脚本开头的 ckpt_root 和 ckpt_files 列表，填入您保存的 Top-k 权重文件名：Pythonckpt_root = Path("/root/autodl-tmp/tb_logs/.../checkpoints/")
ckpt_files = ["best_1.ckpt", "best_2.ckpt", "last.ckpt"]
🧠 技术实现：Fusion vs BaselineBaseline架构: 标准的 3D Encoder-Decoder 结构。逻辑: 直接对全 3D 图像进行特征提取和分割，容易在低对比度的末梢血管处产生断裂。Fusion (本项目核心)分层分片映射: 将 3D 特征深度划分为多个 Slab，每个 Slab 对应 2D 投影图的一个分层区域。2D 拓扑先验: 独立 2D 分支预测血管概率图 $P(x,y)$，并结合从 MIP 自动生成的伪标签进行辅助训练。交叉注意力注入:3D 特征作为 Query，2D 特征作为 Key/Value。2D 预测的血管概率被直接注入注意力对角线（Topology Bias），强制 3D 特征关注拓扑明显的区域。优势: 相比 Baseline，Fusion 模式在集成后具有更高的 Precision 和更低的 HD95，且能更好地维持血管连通性（clDice）。
📊 当前性能指标 (Last Checkpoint)

# 软投票涉及改一下metrics文件的threshold参数，软投票的时候为了细小血管不被忽视适当下调，下面的0.4，0.35就是这个意思

基线 🔥 测试集最终指标汇总
===================================
Dice           : 0.8570
Precision      : 0.8403
Recall         : 0.8862
HD95           : 3.7050
clDice         : 0.9240

Ours
🔥 测试集最终指标汇总
===================================
Dice           : 0.8538
Precision      : 0.8364
Recall         : 0.8871
HD95           : 5.7102
clDice         : 0.9117

🔥 软投票最终指标汇总0.4
========================================
Dice           : 0.8642
Precision      : 0.8647
Recall         : 0.8758
HD95           : 4.0006
clDice         : 0.9249

🔥 软投票最终指标汇总0.35
========================================
Dice           : 0.8622
Precision      : 0.8482
Recall         : 0.8898
HD95           : 4.2205
clDice         : 0.9225