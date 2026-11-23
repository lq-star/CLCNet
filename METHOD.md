# CLCNet 方法说明（相机–LiDAR 外参在线标定）

## 定位

面向无人驾驶/机器人的**相机与激光雷达外参在线标定**：在随机扰动外参下，由网络预测相对位姿残差，并用置信度门控决定是否更新，支持多步迭代精修。专利点相对 LCCNet 在于「置信度估计 + 门控迭代 + 在线拒绝错误更新」。

## 相对 LCCNet 的完善

| 模块 | LCCNet | CLCNet |
|------|--------|--------|
| 主干 | 双编码器 + Cost Volume | 同（可热启动 LCCNet 权重） |
| 输出 | Δt, Δq | Δt, Δq, **confidence** |
| 迭代 | 固定多档级联 | **置信度软/硬门控**多步精修 |
| 损失 | CombinedLoss | CombinedLoss + **conf BCE** + **gate penalty** |
| 在线 | 每步都更新 | 低置信度时**拒绝更新**，避免误差放大 |

## 训练配方 P0（冲 LCCNet，`bash train_p0.sh`）

上一轮全网上大 lr 微调会破坏 LCCNet 主干。P0 改为：

1. **每一档**用对应 `vod_iter{k}.tar` 初始化（不是只级联上一档 CLCNet）  
2. 先 **冻骨干只训置信度头**，再 **backbone_lr = 0.05 × lr** 解冻  
3. `train/eval_iters=1`，按 LCCNet 单档口径用 **t_med** 选模  
4. Epoch：`30/30/30/40/40`，freeze：`12/12/12/15/15`；batch 160  
5. 输出：`outputs/checkpoints/vod_p0/`

结果（VoD val）：单档 ±0.1 m/1° ≈ 6.15 cm / 0.54°（LCCNet 6.08 / 0.47）；五档迭代 ≈ **9.30 cm / 1.16°**（LCCNet **8.61 / 1.00**）。粗档平移已优于 LCCNet，细档旋转略差拖累级联。

## 训练配方 P1（冲超 LCCNet，`bash train_p1.sh`）

针对 P0「细档旋转变差 → 迭代掉点」：

1. 仍逐档 LCCNet quality 初始化  
2. **更长冻骨干** `20/20/22/30/30`，**backbone_lr = 0.02 × lr**，总 lr `2e-4`  
3. **全点云** CombinedLoss（`max_points=0`），batch 120  
4. 选模 **tr_med = t_med + 10×r_med**（护旋转）  
5. 细档 soft multi-step：`train/eval_iters = 1/1/2/2/3`  
6. Epoch：`40/40/45/60/60`；输出 `outputs/checkpoints/vod_p1/`  
7. 训完自动跑 `evaluate_quality.py`

结果：单档 ±0.1 m/1° **超过** LCCNet（5.68 / 0.48 vs 6.08 / 0.47）；但五档迭代 **9.51 / 1.10**，仍落后 8.61 / 1.00。主因是 iter3/4 用 multi-step 选模，与级联「每档只推 1 次」错位，单步 per-range 变差。

## 训练配方 P2（冲迭代，`bash train_p2.sh`）

1. **全程 `train/eval_iters=1`**，选模与五档 cascade / per-range single 对齐  
2. **复用 P1 已强档** iter1/2/5；**只重训薄弱 iter3/4**（从 LCCNet quality 初始化）  
3. iter3/4 更长：epoch `70/80`，freeze `40/45`，`backbone_lr=0.015×lr`  
4. `tr_med` 且 **r_weight=15**；全点云；batch 120  
5. 输出：`outputs/checkpoints/vod_p2/`，训完自动评估  

目标：五档迭代 median **≤ / < 8.61 cm**。

## 专利要点（交底可写）

1. 基于视觉–激光深度图匹配的外参残差回归网络  
2. 同步输出标定置信度，用于门控更新与早停  
3. 软门控训练 / 硬门控推理的在线迭代精修流程  
4. 置信度监督：由当前位姿误差指数映射为伪标签  
5. 多扰动量程课程学习，支持从粗到精在线标定
