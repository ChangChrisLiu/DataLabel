# SAM 2.1 vs SAM 3（及 DINOv3）对比实验报告

- 日期：2026-09-18；执行：Opus worker；核对与结论：Fable
- 代码：`experiments/sam_compare/`；输出：`D:\DataSet\experiments_out\sam_compare\`（`summary.md` 全部聚合表、两份逐掩码 CSV、`concepts.csv`、`dinov3_probe.json`、`model_agreement.json`、`overlays/` 40 张三联图）

## 1. 设置

- 参考掩码：HumanSignal 旧标注（扫描仪视角），分层抽样 **1,495 个掩码、37 个标签、287 帧、14 台机器**；跳过 P_0 被替换的帧。参考是质量不一的人工多边形，不是完美真值。
- 协议（与标注工具一致）：以目标为中心取 512–1024 px 的**原分辨率**裁剪窗 → 模型推理 → 贴回。提示三种：框（参考框外扩 2 px）、单点（距离变换最内点）、点 + 框。
- 模型：SAM 2.1 Hiera-L；SAM 3（交互式图像预测器）。指标：IoU、2 px 容差边界 F 值、耗时。

## 2. 主结果

| | 框 | 单点（取最佳候选） | 点 + 框 |
|---|---|---|---|
| SAM 2.1 IoU | 0.744 | 0.410（0.626） | **0.759** |
| SAM 3 IoU | 0.737 | 0.420（0.631） | 0.753 |

| 点 + 框 | SAM 2.1 | SAM 3 |
|---|---|---|
| 小件（< 400 px）IoU / 边界 F | 0.674 / 0.779 | 0.673 / 0.787 |
| 螺丝 IoU / 边界 F | 0.748 / 0.754 | 0.754 / 0.783 |
| 大件（> 20k px）IoU | 0.883 | 0.875 |
| 图像嵌入耗时 | 27 ms | 43 ms |
| 显存 | 1.3 GiB | 3.9 GiB |

**两个模型打平。** SAM 3 在小件边界上略好（边界 F +0.01–0.03），整体 IoU 略低，嵌入慢 1.6 倍、显存多 2.7 GiB。

## 3. 关键发现

1. **小件 IoU 的天花板是参考标注的误差，不是模型。** 两个模型彼此之间的 IoU 为 0.877（小件 0.864），而各自对人工参考只有约 0.77（小件 0.69）。叠加图可见：扫描仪上螺丝直径只有约 15 px（面积约 170 px），人工多边形常画成偏大的圆环，1–2 px 的边界差异就会让 IoU 掉到 0.7。
2. **单点提示对大件不可靠**（> 20k px 时 IoU 0.24），但三个候选里的最佳候选有 0.73——是候选选择问题而不是分割能力问题。
3. **SAM 3 文字提示不能当主力**：输入 "screw" 全图召回只有 0.37，2×2 分块后 0.62（同时产生 2,938 个候选、耗时 4 倍）；"RAM module"、"power supply" 全图零检出。
4. **DINOv3 冻结特征 + 线性探针**：patch mIoU 0.348（主板 0.67，螺丝 0.05，接口 0.10）。16 px 的 patch 对 20–30 px 的螺丝天然不匹配；作为辅助模型骨干没有优势。

## 4. 结论与对标注工具的决定

1. **交互分割保留 SAM 2.1**，不换 SAM 3（无精度收益，成本更高，Windows 上接口更脆）。
2. **默认提示用"点 + 框"**：倒序流程里差分热图的变化斑块本身就给出一个框；用户点击提供点。单独点击时展示 **3 个候选**供切换（`multimask=True`），大件提供拖框工具。
3. 裁剪窗用 512–1024 px 原分辨率（与现有实现一致）。
4. **训练式辅助模型继续用 YOLO26-seg / RF-DETR-Seg**，不转 DINOv3 骨干；SAM 3 文字提示仅可作为首帧螺丝提议的可选补充（P2）。
5. 对数据集本身的含义：扫描仪视角下螺丝类小件的掩码 IoU 对 1–2 px 误差极敏感，论文里小件应同时报告**检测框指标与边界 F 值**，并在双标一致性里单独统计小件；精细的小件掩码更适合在 OAK1（分辨率约为扫描仪的 2 倍）上评测。

## 5. 复现

```bash
source D:/DataSet/envs/activate_tda.sh
D:/Anaconda/envs/tda/python.exe -m experiments.sam_compare.prepare_sample
D:/Anaconda/envs/tda/python.exe -m experiments.sam_compare.run_interactive --model sam2.1
D:/Anaconda/envs/tda/python.exe -m experiments.sam_compare.run_interactive --model sam3
D:/Anaconda/envs/tda/python.exe -m experiments.sam_compare.aggregate
```
