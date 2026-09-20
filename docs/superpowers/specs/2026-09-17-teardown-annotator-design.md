# 桌面拆解数据集标注工具（Teardown Annotator，TDA）设计文档 v1.6

- 日期：2026-09-17
- 状态：**已按用户回答与 Opus 讨论修订（Fable 逐条裁定）；进入实现计划阶段**
- 分工：Fable（编排核心：设计、监督、验收）+ Opus 5 workers（实现）；Fable/Opus 各完成过评审讨论
- 目标会议：CVPR 2027（截稿 2026-11-16 AoE）

---

## 0. 摘要

构建一个运行在本工作站（RTX 5090 / 63 GB / Windows 11）上的 **PySide6 本地一体化标注工具**，为 66 台台式机拆解序列生成：实例掩码（可见 + 近似完整形状）、零件状态、步骤动作、物理约束图，并程序化导出 COCO 与 VLM 训练/评测数据。核心思路：

1. **倒序 + 图层模型**做传播：从拆解末帧往回标，每步只"装回"发生变化的零件，下层可见区域自动计算。
2. **真值是逐帧物化并人工确认的掩码表**，图层模型只是生成器；后续修改绝不静默覆盖已确认的帧。
3. **四个视角各自从头标注，视角内保持一致**（v1.5）：身份、状态机、约束来自步骤表，四个视角共用；形状、层级、位姿段、遮挡每个视角自己的一套，用同一个倒序流程画。扫描仪仍是最先标、最清晰的视角，但**不假设**它的标注能自动变成其他视角的标注——遮挡差别太大，实测也不支持（见 §2.5）。
4. **结构化标注是唯一事实来源**，VLM 文本全部由它生成，答案可程序校验。
5. **约束 = 物理必要性**，由属性规则 + 机型族模板生成、人工确认；实际拆解顺序只用于校验。
6. **先把 66 台全部标完，再划分数据集**；尽早训练模型辅助标注。

### 0.1 已与用户确认的决定

| # | 决定 |
|---|---|
| C1 | 不再使用 HumanSignal Label Studio；做本地一体化工具；PySide6 桌面程序，核心逻辑与界面解耦 |
| C2 | 标注人力：用户 + 1 位同学，轮流在本工作站使用；标注时间约每天 6 小时 |
| C3 | **四个视角都要标注**：扫描仪（主视角）、OAK 相机1、OAK 相机2、RealSense |
| C4 | 默认从后往前标注，采用图层模型 |
| C5 | 模型输出必须可直接精细编辑，禁止"删掉再推理"式流程 |
| C6 | 精度优先：先标一部分 → 训练模型 → 模型辅助加速；越快进入辅助阶段越好 |
| C7 | 实例 = 记录中被操作过的零件 + 机箱本体；线缆不作为实例（遮挡处从形状中擦除）；接口实例只标插头本体；子零件随父零件一起消失 |
| C8 | 机器人数据独立，本次不涉及 |
| C9 | 已验证：第 k 步图像 = 执行完动作 k 之后的状态（第 1 步为初始状态）；Dell 散热器为不脱落螺丝（拧松是状态变化） |
| C10 | 扫描仪每步使用 `P_0.png`（与用户此前上传 Label Studio 的图一致；`ext.py` 规则） |
| C11 | 约束按物理必要性定义；用途 = 拆解顺序规划 + VLM 对当前状态的理解（有什么、什么现在能拆、什么不能） |
| C12 | 统一词表与用词 |
| C13 | 关系集合要**完整但不冗余**：宁可多记，避免事后补标 |
| C14 | VLM 定位 = 理解 + 推理；数据集设计要以"如何增强 VLM 在此任务上的能力"为出发点；benchmark 用其他 VLM 基础模型 |
| C15 | **先标全部 66 台，不预先划分**；划分放到最后 |
| C16 | 桌上已拆下的零件：**可见就标**，并标为 `detached`（类别不变），供检测/分割与 VLM 推理使用 |
| C17 | 人力细节（Q10）暂缓；先把标注软件做出来 |

### 0.2 仍待确认（已填入默认方案，可在试标后再定）

| # | 问题 | 默认方案 |
|---|---|---|
| **R1** | RealSense 视角的质量等级 | **bronze**：模型预标（平面小零件用单应中心点提示，大件用模型）+ **帧级**人工验收（这帧整体对不对），不逐实例确认；最短边 < 6 px 的实例标 `too_small`（只保留中心点），6–12 px 标 `visible_tiny`（只导框）；导出时按视角给类别白名单 |
| **R2** | 桌面零件的标注范围 | 只在能看到堆放区的视角（主要 OAK2）标；**P0 为框 + 类别 + `placement`**（倒序下每步只是"少一个框"，成本很低），掩码由 SAM 从框生成、P1 再核验；不承诺完整形状；每帧记录 `bench_annotated`，未标帧的堆放区在训练时作为忽略区 |
| **R3** | 每个动作是否记录难度（1–5）与失败原因 | 记录：难度一键打分（默认继承模板），失败原因选枚举 + 可选备注（§3.1） |
| **R4** | 数据冻结日 | 2026-11-03（划分在冻结时确定）；若 10/9 检查点显示进度不足，可议推迟到 11/6 |
| **R5** | 约束边的必要性分级 | 每条边带 `necessity ∈ {required, recommended}`，默认 required；评测同时报"严格/宽松"两组数。用户定义约束为物理必要，因此 recommended 只在标注者明确犹豫时使用 |
| **R6** | 扫描仪全部机器 gold 的目标 | 目标仍是 66 台 gold；**10/9 检查点**若单台耗时降不到 1.5 h 以下，默认切换为"45 台 gold + 21 台 silver（步骤/状态/约束齐全，掩码为模型初稿）"——VLM 标签规模不受影响 |

---

## 1. 目标、范围与优先级

### 1.1 目标

- **CV**：四视角实例分割/检测（23 类 + 属性），含小零件（螺丝、卡扣、接口）；状态识别；`detached` 零件识别。
- **约束**：每台机器一张带物理原因的约束图，可计算任一状态下的合法动作集合。
- **VLM**：理解（看到什么、状态、已完成的历史、进度）+ 推理（刚做了什么、现在能做什么/为什么不能、下一步、剩余计划、计划验证、跨视角一致性）。
- **可报告的质量**：逐帧确认率、双标一致性、模型辅助效率（初稿 vs 终稿）。

### 1.2 范围

**v1 包含**：66 台机器 Disassemble 序列（约 2,830 步）× {扫描仪, OAK1, OAK2, RealSense}。

**v1 不包含**：Components / Side view 文件夹、房间视频、128 张单反照片、机器人数据、网页界面、多人同时在线。

### 1.3 优先级

| 级别 | 内容 |
|---|---|
| **P0** | 扫描仪 gold（66 台）；步骤/动作/状态；约束图；模型训练闭环；OAK1 gold（先测试机型族）；OAK2 silver（含堆放区框）；RealSense bronze；COCO 导出；VLM P0 任务 + 零样本评测；**9/30 前一次端到端演练**（D13：标注 → COCO → VLM JSONL → 零样本评测） |
| **P1** | OAK1 全量 gold；OAK2 堆放区掩码核验；VLM P1 任务、人类基线、微调一个开源 VLM；双标一致性 |
| **P2** | OAK2 全量精修；OAK1 RGB-D 对齐导出；LLM 改写问答；SAM 3 概念提示；多模型预标对比；SAM 解码器微调；超像素贴边 |

---

## 2. 数据底座

### 2.1 源数据

| 视角 | 源路径（F: 只读） | 每步使用的文件 |
|---|---|---|
| 扫描仪 `scan` | `UGA DATA\TAMU_B2.3_*_RGB\...\<desktop>\RGB<step>1\P_0.png`（1600×1600） | `P_0.png`（曝光异常时按 §2.4 改选） |
| OAK 相机1 `oak1` | `OAKD Capture\...\Desktop N\Disassemble\Camera_1\NNN\*_rgb_12mp.jpg`（4032×3040） | 12MP 图；记录 `rgb_aligned.png`、`depth_raw.npy` 路径供导出 |
| OAK 相机2 `oak2` | 同上 `Camera_2` | 同上 |
| RealSense `rs` | `Realsense Capture\Dataset Information\Exp\Desktop N\Disassemble|Disassembly\NNN\original_color.png`（1280×720） | 彩色图；记录 `depth_raw.npy` 供 RGB-D 导出 |
| 拆解记录 | Google Drive 每台一张表，正式导入时存入 `D:\DataSet\raw_logs\` | 步骤名、dupli、工具、机型元数据 |

### 2.2 统一索引

- **帧键**：`(desktop_id, step_k, view)`；`step_k` 是**逻辑步**，以 OAK 相机1 拍摄时间戳顺序为准。
- **对齐规则**：OAK1/OAK2 按文件名时间戳配对；扫描仪按 `RGB<step>1` 编号对齐；RealSense 按文件夹编号 + 文件时间对齐；Drive 记录按行序（含 dupli）对齐。
- **已知异常修复规则**（写成 `configs/index_fixes.yaml`，可审阅）：
  - D60：相机2 文件夹错位；相机1 第 003/004 步时间顺序颠倒；
  - D42：相机2 016/017 对调；RealSense 缺 1–12 步；
  - D24：相机2 末步被放入 Components；
  - D10：第 1 步重拍（保留后一次）；
  - 扫描仪：D1 缺 10 步、11 台缺末步、约 50 个步骤只有 9 张连拍；
  - RealSense：D18–22 文件夹名为 `Disassembly`；D1 为 39 步；
  - D66 仅 13 步。
- **记录与相机步数不一致**（D1 −1、D43 −1、D56 跳号 41、D63 11–13 重号）：进入"待人工确认"清单，在步骤表界面逐条处理。
- **缺帧**：某视角缺某步时该帧标记 `missing`，但逻辑步、状态与形状锚点照常存在（§4.2）。

### 2.3 步骤表导入

- 每行生成 `Step`（原始名、dupli、原始工具、备注）。
- 从原始名解析出 0..n 个 `Action` 草稿：目标（实例或虚拟节点）、动词、工具（§6 映射）。
- 步骤类型：`initial`（第 1 步，0 动作）、`dupli`（0 动作，整帧复制）、`compound`（≥2 动作）、`failed`（有动作、无状态变化）、`auxiliary`（挪开障碍物以露出目标，目标可以是虚拟节点 `cable:*`）、`reorient`（翻转机箱，触发位姿断开）、`ignore`（如 "all components (final layout)"）。

### 2.4 本地缓存、选图与 ROI

- **缓存**：所需图像复制到 `D:\DataSet\cache\<view>\D<nn>\s<kkk>.<ext>`（约 27 GB）。扫描仪 1600² 与 RealSense 1280×720 直接使用；OAK 12MP 生成 ROI 裁剪缓存与金字塔。
- **扫描仪选图**：默认 `P_0`；自动检查曝光（均值/饱和比例）与相对连拍中值的偏差（手入镜），异常时改选连拍中最接近中值且清晰度最高的一张，并记录；可人工改选。
- **ROI**：每台机器每个视角有**机箱 ROI**（自动建议 + 人工确认），以及可选的**桌面 ROI**（放置已拆零件的区域，主要 OAK2）。**掩码一律按原图坐标存储**，ROI 只是显示与推理窗口。

### 2.5 几何

- **帧内配准（同视角相邻步）**：
  - **MVP（扫描仪）**：恒等变换；机箱被翻转由步骤类型 `reorient` 或人工标记触发位姿断开。
  - **P1**：RANSAC 估计相对位姿段参考帧的相似变换，超过阈值（扫描仪 2 px、OAK 3 px、RealSense 1 px）才应用；**不做链式累积**；残差可视化给人确认。
  - 位姿断开：开启新位姿段，该段形状重新绘制，实例 ID 沿用。
  - **位姿段按视角独立（v1.5）**：断点 = `reorient` 步（四视角共用）∪ 该视角自己的**位姿断点**（`pose_break`，人工添加或由审计结果提议后人工接受）。相机被碰只影响它自己的视角。实测（`experiments_out/plan_b_probe/camera_moves/report.md`，桌面固定标记 + 逐事件目检）：OAK1 12 个稳定时段（序列中途 5 次：D2、D4、D29、D32 为 2–9 px，D36 第 18→19 步约 190 px；D20→D21 重新取景），OAK2 4 个（D1 第 33→34 步、D3 后、D31 后），RealSense 与扫描仪 10 天从未移动；扫描仪视角的机箱移动：D63 第 31→32 步旋转约 90°、D64 第 15→16 步、D36 第 40→41 步 13 px、D61 第 1→2 步 18 px。
- **跨视角（v1.5 改写）**：**不做标注迁移。**每个视角在自己的图像上用"任务卡（身份来自日志）+ 本视角帧间差分（定位）+ SAM/人工（形状）"完成；种子机器之后再加本视角训练的检测器出候选。实测依据（`experiments_out/plan_b_probe/transfer/report.md`，D13/24/33 四视角旧多边形作真值）：4 角点单应的投影误差在 12 MP OAK 图上为 34–200 px（视差主导，点击精度不是瓶颈：角点抖动 σ=12 px 命中率变化 < 4 个百分点），小于约 40–60 px 的零件命中 0%，框 IoU 0.17–0.46。4 角点单应、AprilTag 外参（36h11 板，44/66 台有，扫描仪没有）、OAK 深度三维迁移都只允许作为**可选的粗提示**（大件、误差圈），任何流程、排期、质量等级不得依赖它们。
- **OAK 12MP ↔ 1280×800 对齐图（P2）**：逐台估计"缩放 + 裁剪"并检验是否恒定；导出 RGB-D 掩码时使用，截断帧标记。RealSense 彩色图与深度图本身已对齐。

---

## 3. 数据模型

### 3.1 实体

| 实体 | 关键字段 |
|---|---|
| `Desktop` | id、品牌、机型族、机箱平台、机箱类型、尺寸、日期、划分（最后填）、备注 |
| `Instance` | `instance_key`（如 `screw.cpu_cooler.03`）、类别（§6）、属性（role、head、head_source、kind、captive…）、`parent`（随哪个零件一起移除；`attached=true` 时连带）、`mounted_on`（物理支承：RAM→主板、主板→机箱、硬盘→硬盘笼）、`fastens`（仅螺丝：固定的是哪个零件）、`slot_id`（机型族模板槽位）、`group_id` + `group_order ∈ {unordered, sequential, opposite_pairs}`（同父同类集合的顺序约束，默认 unordered）、`placement ∈ {in_chassis, on_bench, elsewhere}`（物理位置，随 `remove` 事件默认变为 on_bench）、`removal_direction`（机箱坐标系，默认来自模板）、原始名列表 |
| `Connector`（`Instance` 子类型） | `socket_host`（插在哪个实例/虚拟节点上）、`cable`（所属虚拟线缆节点） |
| `VirtualNode` | 不画掩码的节点：`chassis_front_panel`、`cable:<描述>`（属性 `owner`：PSU / 前面板 / 机箱风扇 / CPU 风扇 / 独立；状态 routed / released）等 |
| `Step` | k、类型（§2.3）、原始名、dupli、备注、时长（由时间戳计算） |
| `Action` | step、目标（实例或虚拟节点）、动词、工具、方向、结果（success / failed）、`failure_reason`（枚举：blocked_by_cable / blocked_by_part / fastener_stuck / wrong_tool / other + 备注）、`difficulty`（1–5，默认模板值） |
| `StateEvent` | 目标、step、旧值→新值、`evidence_view`（在哪个视角判定，自动记录）；由 `Action` 自动生成（含父零件移除时附着子零件的连带事件、`placement` 变化），可手工补 |
| `TrackingPolicy` | 类别 × 状态 × placement → 是否需要掩码/框（默认见 §6.2），可按实例覆盖 |
| `Frame` | 帧键、源文件、位姿段、变换 `T_k`、标记（`hand_or_tool_in_frame`、`in_progress`（动作未完成的瞬态）、`image_quality`（四个视角都填）、missing）、`bench_annotated`（堆放区是否已标）、扫描仪连拍逐张质量指标 `burst_metrics[10]` 与选中理由、审核状态（未标 / 自动 / 已确认 / 需复核） |
| `PoseSegment` | 视角、起止步、参考帧、**4 个带语义的机箱角点**（前上 / 后上 / 后下 / 前下，机箱坐标系）→ 到扫描仪的单应 + 机箱 X/Y 轴在图像中的方向 `chassis_frame_in_image` |
| `ShapeKeyframe` | 实例、视角、位姿段、`anchor_step`（该形状适用的最晚逻辑步）、`placement`（in_chassis / on_bench）、几何类型（mask / box）、形状部件列表（每部件一张 RLE 或框，参考帧坐标）、`amodal_complete`、来源（手画/SAM/模型@版本）、初稿引用、耗时与编辑次数 |
| `ZOrder` | (视角, 位姿段) 下 (实例, 部件) 的全序，带版本号；堆放区实例单独一组，组间不排序 |
| `PairOverride` | (视角, 位姿段, A, B)：A 在 B 之上，优先于全序 |
| `OccluderMask` | 帧级遮挡层，`occluder_type ∈ {hand, arm, body, tool, cable, other}`，只作用于当前帧 |
| `FrameOverride` | (实例, 帧)：替换可见掩码，或设置 `visibility`（见 §6.2）。**所有"只改这一帧"的编辑都写到这里** |
| `InstanceFrameFlags` | (实例, 帧)：`difficulty_flags`（低对比 / 反光 / 极小 / 形状歧义 / 类别歧义，可选一键） |
| `CompiledMask` | 真值表（§3.4），含派生的 `visibility` 与 `occlusion_ratio` |
| `Relation` | §7；每条边带 `necessity`、`mode`（仅 blocked_by）、`reason`（manual 边的自由文本） |
| `Op` | 源操作日志（撤销/重做、审计） |

### 3.2 实例身份

- **身份由步骤表决定**：`instance_key` = 类别 + 角色 + 同类序号（按首次被操作的逻辑步排序），不依赖记录中的编号。
- **图像中"哪一颗是 3 号"**：MVP 在扫描仪上叠加第 k−1/k 帧差分热图由人点选；P1 起自动把本步目标匹配到变化斑块。不脱落螺丝差分弱时人工点选。
- **四视角同一身份**：第 k 步变化的实例在所有视角是同一物体——身份由任务卡给出，不靠几何。同类多实例（4 根内存、6 颗主板螺丝）靠**时间**区分：哪一步消失的是哪一个，由该视角自己的帧间差分指出（v1.5：不再用跨视角单应投影）。

### 3.3 编译器（纯函数、可单测）

对视角 v、逻辑步 k：

1. `state(k)` = 初始状态 + 所有 step ≤ k 的事件（父零件 `removed` 时，`attached` 的子零件自动 `removed`）。
2. `needs_mask(k)` = （`placement=in_chassis` 且 `TrackingPolicy` 要求掩码的实例）∪（`placement=on_bench` 且本视角有堆放区 ROI 的实例，几何类型可为框）。
3. 对每个 `i ∈ needs_mask(k)`：在同一位姿段内取 **`anchor_step ≥ k` 中最小者**对应的关键帧（关键帧 j 适用于 (上一个锚点, anchor_j]）。若无 → `missing_shape`：机箱内实例阻塞该帧确认；堆放区实例只产生提示并使该帧 `bench_annotated=false`。
4. 形状经 `T_k`（最近邻）变换到帧坐标 → `A_i`。
5. `V_i = A_i − ∪{A_j : j 在 i 之上}`，逐对判断：有 `PairOverride` 用它，否则用 `ZOrder`。多部件实例按部件分别参与；桌面上的零件与机箱内零件不重叠，层级只在彼此堆叠时起作用。
6. `V_i = V_i − OccluderMask_k`；若有 `FrameOverride`，以其为准。
7. 输出 `V_i`、`A_i` 引用、**遮挡比例** `1 − |V_i|/|A_i|`、派生的 `visibility`（无覆盖时：遮挡比例 < 0.3 → visible，0.3–0.95 → occluded_partial，≥ 0.95 → occluded_full；最短边 < 6 px → too_small，6–12 px → visible_tiny；`FrameOverride.visibility` 优先）与**输入哈希**。

**关键帧与标注方向**：
- 实例寿命由步骤表预先确定，因此首次绘制时默认锚点 = 该实例在本位姿段内最后一个需要掩码的逻辑步。
- 一个实例可以有两条关键帧链：`installed` 链（移除步之前）与 `detached` 链（移除步及之后），锚点机制自然区分。
- 在第 j 步"拆分关键帧"时，新形状作用于**当前浏览方向**一侧：倒序时新关键帧锚点 = j；正序时原关键帧锚点改为 j−1、新关键帧继承原锚点。

### 3.4 真值表 `CompiledMask`

| 字段 | 说明 |
|---|---|
| 键 | (desktop, view, step, instance) |
| `visible_rle`、`occlusion_ratio`、`placement` | 可见掩码（原图坐标）、遮挡比例、installed/detached |
| `status` | `auto`（随输入刷新）/ `verified`（整帧确认后冻结） |
| `input_hash`、`verified_by`、`verified_at` | 溯源 |

规则：

- 输入变化时，只刷新 `auto` 行。
- 冻结行的重编译结果与冻结值比较：**容差对称差**（忽略边界 2 px 以内的差异）> max(2% 面积, 20 px) → 生成冲突，进入冲突队列（保留旧值 / 接受新值 / 编辑），**绝不静默覆盖**。
- 向已确认帧新增或删除实例 → 该帧降为"需复核"。
- 导出只读 `CompiledMask`，报告 verified 比例与质量等级。

### 3.5 存储

- SQLite（WAL 模式）：`D:\DataSet\annotations\tda.sqlite`；RLE 为 pycocotools 格式。
- 备份：每日与每次退出时用 SQLite backup API 备份到 `F:\PHD Data Backup\Desktop Dataset\TDA_backups\`（14 天滚动 + 每周快照）。
- 单用户锁文件（当前标注员与时间）。

---

## 4. 标注流程与界面

### 4.1 每台机器的工作单

| 阶段 | 内容 | 预计耗时 |
|---|---|---|
| S0 自动准备 | 索引、缓存、选图检查、ROI 建议、步骤表草稿、差分热图 | 无人工 |
| S1 步骤与实例核对 | 每行显示扫描仪第 k−1/k 帧缩略图，预填动作/目标/工具/难度；实例表填父零件、`socket_host`、`cable`、角色、头型；失败步填原因 | 已有模板 10–15 min；机型族首台 30–45 min |
| S2 扫描仪倒序标注（gold） | §4.2 | 2–2.5 h（模型辅助后目标 1.5 h） |
| S3 OAK1 倒序标注（gold） | 与 S2 同一流程，在 OAK1 自己的图上从头画：本视角位姿段/ROI → 任务卡 + 本视角差分 → SAM/人工 → 确认。种子机器之后由本视角检测器出初稿 | **待试标实测**（v1.4 的 0.6–1 h 以迁移为前提，已作废；先按与 S2 同量级估计） |
| S4 OAK2 倒序标注（silver） | 同上；另有堆放区 ROI：起点帧画出全部已拆零件的框，倒序每步少一个框；操作者遮挡用遮挡层 | 待试标实测 |
| S5 RealSense（bronze） | 本视角检测器预标（种子机器人工标）+ **帧级**验收，不逐实例确认；最短边 < 6 px 标 `too_small` | 待实测 |
| S6 约束图 | 规则/模板实例化 → 接受/拒绝 → 校验（§7） | 已有模板 10 min；首台 30 min |
| S7 质检与冻结 | 自动检查全部通过、冲突清零、标记完成 | 5–10 min |

**顺序**：所有机器先做 S0–S2 + S6（扫描仪 gold + 约束），按机型族分批（大族先做，模板尽早复用）；约 8–10 台后训练模型；S3–S5 在模型可用后按视角分批进行。

### 4.2 倒序标注（S2，其余视角同理）

1. **起点** = 该视角最后一个可用帧；在起点帧画出所有需要掩码的实例（若该视角有堆放区 ROI，则为堆放区内每个已拆零件画框）。
2. **从第 k 帧退到第 k−1 帧**时，任务卡列出动作 k 引起的变化。**任务卡属于屏幕上的这一帧（k−1）**：卡上要画的形状都画在当前图像上，"已完成"按当前帧判断；差分热图 = 当前帧（k−1）对刚完成的第 k 帧；`Tab` 闪切到第 k 帧（`Shift+Tab` 闪切到第 k−2 帧）。（v1.4 澄清：计划里曾写成"第 k 帧的任务卡描述去往 k−1 的工作"，那样零件在第 k 帧已被拆下，无法在该帧提交。）
   - `removed → installed`：新增机箱内形状，默认放最上层；同时列出随它一起装回的附着子零件；若该零件在堆放区有框，则它从堆放区消失（on_bench 链在 k 结束）；
   - `open → closed`、`unplugged → plugged`、`displaced → installed`：在 k−1 处拆分关键帧；
   - `loosened → fastened`：只改状态，形状沿用；
   - `dupli`：整帧复制，一键确认；`failed`：无状态变化，一键确认（但可标记 `in_progress`/手入镜）。
3. 画布叠加差分热图；点击 → SAM（在原分辨率裁剪窗推理）→ 编辑 → Enter。
4. 处理差分中未被任何事件解释的明显区域：拆分关键帧 / 画遮挡层 / 标记忽略。
5. 空格确认本帧 → 写入 `verified` → 下一帧。

**缺帧处理**：扫描仪缺第 k 步时，S2 跳过该帧掩码，状态与锚点按逻辑步照常；其他视角对无扫描仪对应帧的步回退为原生倒序标注。

### 4.3 编辑语义

每次像素编辑必须明确作用范围，画布下方显示"影响 N 帧"缩略条：

| 范围 | 含义 | 默认触发 |
|---|---|---|
| **改形状关键帧** | 修改当前生效的关键帧，影响其适用区间内所有未冻结帧 | 编辑像素不落在其他实例形状内 |
| **改层级** | 写 `PairOverride` | 画笔加像素落入 B 的形状 → 提示"把 A 放到 B 之上？"；橡皮擦掉 A 中与 B 重叠的像素 → 提示"把 B 放到 A 之上？" |
| **单帧覆盖** | 写 `FrameOverride`，只改这一帧 | 按住 `Alt` |
| **拆分关键帧** | 从当前步起按浏览方向生成新版本 | `Ctrl+K` |

- 画笔/橡皮/顶点编辑**绝不触发模型重推理**。
- 会影响已冻结帧的修改，先提示"将产生 N 个冲突"。

### 4.4 审阅模式

- 逐帧浏览（上下键），显示可见掩码轮廓、状态标签、问题列表；Enter 接受整帧、`R` 标记返工。
- 入口：冲突队列、需复核队列、缺形状队列、未解释变化队列。
- silver 视角的核验（S4/S5）主要在此模式完成：模型初稿 → 逐帧接受/修正。

### 4.5 界面布局

- **顶部**：机器选择；视角切换（1–4）；模式（步骤表 / 标注 / 审阅 / 约束 / 质检）。
- **左侧**：纵向时间轴缩略图（灰=未标、黄=自动未确认、绿=已确认、红=冲突/需复核）。
- **中间**：画布——打开即缩放到 ROI；滚轮缩放，>400% 显示像素网格；小地图；掩码填充/轮廓切换与透明度；**`Tab` 在第 k 与第 k−1 帧之间闪切对比**；可并排显示另一视角的同一步作参考。
- **右侧**：任务卡；当前帧实例列表（层级拖拽、显隐、状态标签、`detached` 标记）；属性面板。

### 4.6 编辑工具

| 工具 | 阶段 |
|---|---|
| 画笔 / 橡皮（像素级，`[` `]` 调大小） | MVP |
| SAM 点/框提示（视野内原分辨率裁剪窗，≤1024 边长，超出分块） | MVP |
| SAM 局部修正（以当前掩码为 `mask_input` 加正负点，只接受修正点附近的变化） | MVP |
| 填洞 / 去小连通域 | MVP |
| 遮挡层画笔（`O`） | MVP |
| 撤销 / 重做（`Op` 日志逆补丁） | MVP |
| 多边形顶点编辑（掩码↔多边形，简化容差可调） | P1 |
| 差分斑块自动匹配目标 | P1 |
| 超像素贴边、与邻居布尔运算 | P2 |

---

## 5. 模型辅助

### 5.1 交互式分割（MVP）

- **SAM 2.1 Hiera-L** 图像预测器常驻后台线程；输入为视野裁剪窗（原分辨率）；支持点、框、`mask_input`。
- 显存预算：SAM 约 4–6 GB，其余留给训练。
- SAM 3 概念提示列为 P2 试验。

### 5.2 训练闭环（约 8–10 台扫描仪 gold 后启用）

- v1 只接一个框架：**Ultralytics YOLO26-seg**。
- 训练数据：已确认的 ROI 裁剪 + 可见掩码（类别 = 23 类；`detached` 作为附加属性，训练时可选并入类别）；每新增约 10 台重训，子进程运行，限制批大小避免与 SAM 争显存。
- 用途：扫描仪起点帧提议；其他三视角的批量初稿（预测框 → SAM 精修边界 → 可编辑初稿）。
- **视角冷启动**：每个新视角先人工完成 3–4 台（任务卡 + 本视角差分 + SAM），加入训练集后再对该视角批量预标（跨视角域差大；v1.5：不用单应提示）。
- 多框架对比（RF-DETR-Seg、EoMT/Mask2Former）只作论文基线。

### 5.3 效率记录

- 每个形状保存初稿与终稿，记录初稿 IoU、编辑操作数、耗时。

### 5.4 精度目标

- 边界误差：扫描仪 ≤ 2 px，OAK ≤ 3 px，RealSense ≤ 1 px（大零件）；`too_small` 实例不计。

---

## 6. 统一词表

### 6.1 类别（23 类）

| 大类 | 类别 `class` | 主要属性 |
|---|---|---|
| 结构 | `chassis` | —（不移除，可多部件） |
|  | `drive_cage` | of（ssd / hdd / optical）；记录中的 shield / cage / case 映射到此类 |
|  | `cover` | of（cpu_cooler / motherboard_screws / ram / expansion_slot / front_bezel / drive / other） |
|  | `cooler_bracket` | 散热器固定底座/支架 |
| 主要部件 | `motherboard` | — |
|  | `cpu` | — |
|  | `cpu_cooler` | kind（fan / heatsink / heatsink_fan） |
|  | `ram_module` | — |
|  | `psu` | — |
|  | `storage_drive` | kind（hdd / ssd / unknown） |
|  | `optical_drive` | — |
|  | `expansion_card` | kind（gpu / wlan / other） |
|  | `case_fan` | — |
|  | `misc_part` | name（如 front_io_module） |
| 紧固件 | `screw` | role（motherboard / cpu_cooler / cooler_bracket / drive / optical_drive / card / psu / other）、head（PH1/PH2/PH3/T15/T20/unknown）、captive（bool） |
| 锁扣 | `ram_latch` | of（ram 实例） |
|  | `cpu_socket_lever` | — |
|  | `psu_latch` | — |
|  | `drive_latch` | of（drive_cage / drive 实例） |
|  | `card_latch` | of（card 实例） |
|  | `cooler_latch` | — |
|  | `cable_clip` | holds（虚拟线缆节点） |
| 接口 | `connector` | kind（atx_24pin / cpu_power / sata_data / sata_power / fan / front_panel / usb_header / audio / molex / other）、`socket_host`、`cable` |

### 6.2 状态集与默认追踪策略

| 类别 | 状态（需要掩码的状态加粗） |
|---|---|
| `screw` | **fastened** / **loosened** / removed（桌面可见时可有 `detached` 形状） |
| 各类 latch、`cpu_socket_lever`、`cable_clip` | **closed** / **open** |
| `cover` | **closed** / **open** / removed（同上可 `detached`） |
| `connector` | **plugged** / unplugged（不追踪）/ removed |
| 部件类（含 `drive_cage`、`cooler_bracket`、`misc_part`） | **installed** / **displaced**（画一次新形状）/ removed（桌面可见时可有 `detached` 形状） |
| `chassis` | **present** |
| 虚拟节点 `cable:*` | routed / released（无掩码） |

**帧级可见性 `visibility`**（每实例每帧，自动派生、可人工覆盖，一键快捷键）：`visible` / `occluded_partial` / `occluded_full` / `out_of_view` / `too_small`（只保留中心点）/ `visible_tiny`（只导框）/ `motion_blur`。它同时是 V14"本视角能否判断"任务的真值来源。

### 6.3 动作动词

| 动词 | 适用 | 状态效果 |
|---|---|---|
| `unscrew` | screw | captive = true → loosened；否则 → removed |
| `disconnect` | connector | → unplugged |
| `open` | latch / lever / cover | → open |
| `release` | cable_clip、虚拟线缆节点 | → open / released |
| `remove` | 部件、cover、connector（随线缆整体取出时） | → removed（附着子零件连带 removed） |
| `displace` | 部件（翻开 PSU、推出光驱） | → displaced |
| `reorient` | 机箱 | 位姿断开 |

失败尝试：`result = failed`，不产生状态事件，须填 `failure_reason`。

### 6.4 方向（机箱坐标系）

以机箱立放定义：X = 前→后，Y = 底→顶，Z = 从侧开口向外（垂直主板）。取值：`+Z`、`±X`、`±Y`、`rotate`、`none`。

### 6.5 工具

`PH1`、`PH2`、`PH3`、`T15`、`T20`、`hand`、`none`。

### 6.6 原始名映射

- 以已有分析规则为起点（155 种原始写法 → 52 组 → 23 类 + 属性），写成可审阅的 `configs/taxonomy_map.yaml`。
- 在 S1 步骤表界面显示映射结果；人工改正回写为规则建议。

---

## 7. 关系与字段（完整但不冗余）

### 7.1 显式记录的字段与关系

| 记录位置 | 内容 | 用途 |
|---|---|---|
| `Instance.parent` + `attached` | 生命周期归属：随哪个零件一起移除（不脱落螺丝→散热器） | 连带事件、同组 |
| `Instance.mounted_on` | 物理支承（RAM→主板、主板→机箱、硬盘→硬盘笼、散热器→主板） | 装配结构问答、模板槽位图 |
| `screw.fastens` | 这颗螺丝固定的是哪个零件 | `fastened_by` 规则（不再借用 parent） |
| `Connector.socket_host`、`Connector.cable`、`cable.owner` | 插头插在谁身上、属于哪根线、线固定连在哪个部件 | `connected_to` 规则 |
| `*_latch.of`、`cover.of`、`cable_clip.holds` | 锁扣/盖板/线夹作用对象 | `locked_by`、`covered_by`、线缆 `blocked_by` 规则 |
| `screw.role/head/head_source/captive` | 紧固件属性及头型判读来源（记录 / 图像） | 工具问答（视觉判别版）、连带事件 |
| `Instance.group_id/group_order` | 同父同类集合及其顺序约束（默认 unordered；对角拧松等为 sequential/opposite_pairs） | 顺序合法性校验，避免误判可互换零件 |
| `Instance.slot_id` | 机型族模板槽位 | 跨机泛化分析、模板复用 |
| `Instance.removal_direction` + `PoseSegment.chassis_frame_in_image` | 名义拆卸方向 + 机箱坐标系在图像中的朝向 | 操作方位问答、空间关系程序生成、机器人侧复用 |
| 硬约束边 | `fastened_by` / `connected_to` / `locked_by` / `covered_by` / `blocked_by(mode ∈ {physical_path, tool_access, cable_tension})`，每条带 `necessity ∈ {required, recommended}`、`reason`（manual 边） | 合法动作、顺序规划、推理链文本 |
| `Action.failure_reason`、`Action.difficulty`、`Step.notes` | 失败原因、难度、操作者意图/自述 | 失败案例、难度问答 |

**由其他数据推导、不单独存储**：`occluded_by`（编译器输出，导出时物化）、`same_group`（group_id）、`requires_tool`（head 属性）、空间邻接与相对方位（掩码几何 + `chassis_frame_in_image`）、`supports`/`part_of`/`secures` 等反向或同义边、"合法动作集合"、进度与剩余步数（约束图 + 状态）。**数据库不存"合法动作集合"，但导出的训练/评测 JSONL 必须物化它并写入 `graph_version`，保证评测可复现。**

### 7.2 硬约束语义

对目标执行 `remove` / `displace` / `open`（及对螺丝 `unscrew`、对接口 `disconnect`）之前，另一节点必须处于指定状态：

| 类型 | 所需状态 |
|---|---|
| `fastened_by(X, screw)` | screw ∈ {loosened, removed} |
| `connected_to(X, connector)` | connector ∈ {unplugged, removed} |
| `locked_by(X, latch)` | latch ∈ {open} |
| `covered_by(X, Y)` | Y ∈ {open, removed, displaced} |
| `blocked_by(X, Y)` | Y ∈ {removed, displaced, released} |

**每类边挡住哪些动作（v1.6，取代上面"之前"一句的字面含义；实现见 `tda/core/graph_rules.py`，检查器、合法动作集合、规划器读同一张表）**：

| 类型 | 挡住的动作 | 理由 |
|---|---|---|
| `fastened_by` / `locked_by` | remove、displace、open | 拧着/锁着的零件动不了，但仍可以在它上面拔插头、拧别的螺丝 |
| `covered_by` | 全部 | 被盖住就够不着 |
| `connected_to` | 仅 remove | 线还插着时可以把零件挪开（薄机箱的电源必须先挪开才够得着插头），不能拿走。旧规则"挡全部"在 66 份日志上产生 34 条误报 |
| `blocked_by(mode=cable_tension)` | 仅 remove | 被线拉住：能动，拿不走 |
| `blocked_by(mode=physical_path)` | remove、displace | 挡在路径上 |
| `blocked_by(mode=tool_access)` 或方式缺失/未知 | 全部 | 工具够不着；未知按最保守处理 |

被**拆走**（或不存在）的阻挡物满足任何一条边。`necessity=recommended` 的边是**偏好**：不参与合法性、不参与死锁判定；规划器能遵守就遵守，遵守不了就忽略并标记 `relaxed`（建议只能改变计划的顺序，不能让计划不存在）。关系行的 `status ∈ {proposed, accepted, rejected, accepted_orphan, rejected_orphan}`：规则边不可手改，只能接受/拒绝（存为 `source=override`）；规则不再推导出该边时决定变为孤儿（不生效、可见、可转为手工边），规则回来时决定原样恢复；`rejected` 与两种孤儿状态不生效。`graph_version` 的摘要包含 `status`（审过的图是另一个版本）。S1 的 Apply 在同一事务里用与命令行相同的函数重推本机规则边。

**接口规则**：移除部件 X 之前，所有满足 `socket_host = X` 或 `cable.owner = X` 的接口必须已断开（SATA 数据线两端独立；PSU 线束的所有插头都要求已拔下才能取 PSU）。

### 7.3 生成

1. **属性规则**（预计覆盖约 90%）：`screw.fastens = X` → `fastened_by(X, screw)`；接口规则 → `connected_to`；`latch.of` → `locked_by`；`cover.of` → `covered_by`；`cable_clip.holds` 的线缆若阻挡目标 → `blocked_by(目标, cable, mode=cable_tension)`；固定模板边（如 `cpu covered_by cpu_cooler`、`cpu locked_by cpu_socket_lever`）。
2. **机型族模板**：同族共享"零件槽位图"，实例化到每台机器；每个族的首台人工补全后保存为模板。
3. **人工**：剩余 `blocked_by` 在约束面板添加。

每条边记录：来源（rule / template / manual）、证据（步骤 id）、状态（proposed / accepted / rejected）。

### 7.4 校验

- **图无动作死锁（v1.6，取代"图无环"）**：节点是动作 `(verb, instance)`；动作 `(v, T)` 依赖每条挡住 `v` 的生效必要边 `(T, B)` 的清除动作；清除动作是**一组**候选（凡能让 B 进入该边接受状态的动词，最后再加"拆走 B"——若其类别可拆），对边取"与"、对候选取"或"；死锁 = 这张图里从某个零件的拆除目标可达的环（`graph_plan.find_deadlocks`，全代码唯一的定义；性质：无死锁且无死胡同 ⇔ 每个零件都排得出拆解计划）。零件层面的环不一定是死锁：`fastened_by(支架, 螺丝)` + `blocked_by(螺丝, 支架, physical_path)` 可解，因为"拧螺丝"不被后者挡。面板的任何编辑与 Apply 时的规则重推都不得存入死锁（拒绝并指出环与解法）；命令行对已含死锁的库只报告（退出码 1），不擅自修复。**死胡同**（某条边没有任何清除动作，如 `locked_by(X, 机箱)`）不是错误，但必须可见：页签、待确认问题、报告各列一处。
- 每个成功动作执行前一刻，其前置条件全部满足。
- 每次失败尝试执行前一刻，至少一个前置条件未满足（含虚拟线缆节点），且与 `failure_reason` 一致。
- "合法动作集合"为计算量，不存储。

---

## 8. 导出与 VLM 层

**确定真值原则（v1.6，适用于一切依赖约束图的 VLM 任务）**：约束图记录的是物理**必要**边，可能不完整。因此——图**禁止**的一定做不了；图**允许**的只是上界；日志**演示过**的一定做得了；**状态上已不适用**的（目标已处于该动词的终态、已不在场、已随父件离开）一定做不了，且必须看图才能判断。标签只发布这四类确定的真值，并写明来源 `truth_source ∈ {demonstrated, graph_blocked, state_inapplicable, failed_attempt}`：V4 正例 = 演示过的下一步，反例 = 被图挡住（列全部未满足的必要边）或状态上不适用，按"同动词同类别 → 同动词 → 同类别"匹配采样，约 1:1，指标为按动词宏平均的准确率；V5 的标签是 `{must_include（演示过的）, must_not_include（确定做不了的）, permitted_upper_bound}`，指标为演示召回 + 被挡率，**从不**声称列出了"物理上可能的全部动作"；V6 选项 = 1 个演示过的 + ≥2 个确定错误的 + 标明"未知"的干扰项（种类只在标签侧）；V16 有效计划 = 日志后缀，无效计划 = 违反必要边的篡改（交换、移动、替换），1:1。没有 `graph_version`、没有任何生效边、或含死锁的机器不出这些题；日志含未解析目标（`?`）的机器不出 V10；日志与图矛盾的步骤（当前 9 步 / 15 条边）不作真值。每条记录分 `prompt`（模型可见：task、不透明图像 id、question、options）与 `label`（其余全部）；图像 id 与记录 id 为加盐哈希，映射在标签侧的 `*.images_manifest.jsonl`；导出自带"只看题面"的分类器上界表（动词、类别、动词+类别、模板），V4 各项须 ≤ 65 %。V15 不用任何几何，只用各视角自己已确认的 `visibility`。

### 8.1 CV 导出

- 每视角 COCO 实例分割（可见掩码 RLE；近似完整形状作为扩展字段并带 `amodal_complete`；`detached`、`occlusion_ratio`、状态作为附加属性）；全图与 ROI 裁剪两种坐标版本。
- 帧级状态 JSON、每机约束图 JSON、步骤/动作表。
- 每条标注带质量等级（gold / silver）与确认状态。
- RGB-D：RealSense 直接可用；OAK1 对齐版本为 P2。
- **划分**：在数据冻结时确定，按机箱平台不相交；导出脚本以 `Desktop.split` 为准。

### 8.2 VLM 样本设计

**设计原则**（以"VLM 需要学到什么"为出发点）：

1. **感知落地**：每个答案都能对应到掩码/框（证据可定位）。
2. **状态与变化**：单帧状态、两帧变化、"无变化"负样本（dupli）。
3. **物理因果**：答案附带结构化推理链（观察 → 适用约束 → 结论），可程序生成。
4. **历史与进度**：从单帧推断已完成动作集合与剩余步数。
5. **可见性诚实**：被遮挡/太小/不在视野时，正确答案是"本视角无法判断"；多视角融合时可判断。
6. **合法性而非唯一性**：下一步/计划类问题以约束图给出的合法集合判分。
7. **困难负样本**：非法动作、不存在的零件、失败尝试、近重复帧、桌面上已拆零件与机箱状态的一致性。

**任务清单**：

能力分层：**L1 感知**（V1/V2/V8）→ **L2 可供性**（V4/V5）→ **L3 规划**（V6/V7/V16）→ **L4 元认知**（V14 可判定性/拒答、V15 跨视角）。L1 是任何数据集都能提供的；**L2–L4 依赖"视角无关的状态机 + 多个可见度不同的观测"，是本数据集独有的贡献**，其边际标注成本接近零。

| 代号 | 优先级 | 任务 | 输入 | 答案来源 | 指标 |
|---|---|---|---|---|---|
| V1 | P0 | 看到了哪些组件、在哪 | 单帧 ROI | 可见掩码/框（含 on_bench） | 集合 F1 + 框 IoU |
| V2 | P0 | 零件状态 / 计数 | 单帧 | 状态 | 准确率 / 完全匹配 |
| V3 | P0 | 刚做了什么（动词、目标、工具） | 第 k−1、k 帧 | 动作 | 各槽位准确率 |
| V4 | P0 | 能否对 X 执行某动作、为什么不能（含反事实：现在直接抽 PSU 会怎样） | 单帧（+历史） | 约束图 + 状态 | 是否准确率 + 阻碍集合 F1 |
| V5 | P0 | 现在能做哪些动作 | 单帧（+历史） | 合法动作集合 | 集合 F1（严格/宽松两组） |
| V6 | P0 | 下一步做什么 | 单帧 + 历史 | 合法动作集合 | 合法率 + 与实际匹配率 |
| V8 | P0 | 指代定位 | 单帧 | 可见掩码 | IoU@0.5 / 点在掩码内（tiny 只算点） |
| V10 | P0 | 已完成了哪些动作（历史推断）、进度分箱 | 单帧 | 状态 → 已发生事件；约束图 → 剩余 | 集合 F1；分箱准确率 |
| V12 | P0 | 两帧之间是否有变化（dupli；P1 加入扫描仪同步连拍作负样本） | 两帧 | dupli / 事件 | 准确率 |
| V14 | P0 | 可判定性 / 拒答：本视角能否判断 X 的状态 | 单帧 | `visibility` + 状态是否由本视角不可见特征决定 | 准确率 + risk-coverage / 过度自信率 |
| V15 | P0 | 跨视角一致性：是否同一时刻同一机器；A 视角被挡的 X 在 B 视角是什么状态；哪个视角更适合回答 | 两帧 | 共享实例 + `visibility` | 准确率 |
| V16 | P0 | 计划验证 / 找错：给定候选序列是否合法、第一处错在哪 | 单帧 + 序列 | 约束图（一半为程序生成的错误序列） | 准确率 + 定位 |
| V7 | P1 | 达到目标的剩余计划 | 单帧 | 约束图 | 拓扑合法性 + 达成目标 + 冗余步数 + 第一处违规 |
| V9 | P1 | 紧固件头型视觉判别（十字/梅花）与所需工具 | 扫描仪/OAK1 局部图 | `head`（仅 head_source=图像的样本） | 准确率 |
| V13 | P1 | 真实失败尝试的解释（5 例，作定性案例） | 两帧 | `failure_reason` + 约束 | 案例研究 |
| V17 | P1 | 堆放区零件与机箱状态一致性 | 单帧（OAK2） | on_bench + 状态 | 集合 F1 |
| V18 | P1 | 操作方位（拆卸方向、相对位置） | 单帧 | `removal_direction` + `chassis_frame_in_image` + 掩码几何 | 准确率 |

- 输入图像：扫描仪或 OAK1 的 ROI 裁剪（V17 用 OAK2；V14/V15 覆盖四个视角）。
- **推理链格式**：封闭操作集 `op ∈ {observe, recall_relation, check_precondition, propagate_state, compare_frames, conclude, abstain}`；每个 `observe` 必须带框/点与 `visibility`；记录 `depth`（推理步数）用于按深度分层评测；自然语言由模板渲染。例：
  `{"task":"V4","answer":{"feasible":false,"blockers":["screw.motherboard.03"]},"rationale":{"depth":3,"steps":[{"op":"observe","target":"screw.motherboard.03","value":"fastened","evidence":{"view":"scan","bbox":[...],"visibility":"visible"}},{"op":"recall_relation","edge":["motherboard","fastened_by","screw.motherboard.03"],"necessity":"required"},{"op":"check_precondition","required":["loosened","removed"],"actual":"fastened","satisfied":false},{"op":"conclude"}]}}`
  评测时除最终答案外报告 **推理链落地准确率**（中间步的框是否正确）。
- **困难负样本**：同机相邻两步（只差一颗螺丝）；同平台不同机器的同一步（防止记忆机器指纹）；"合法但当次没做"的动作作为 V6 干扰项；对被遮挡/已移除零件提问（正确答案为拒答或"已移除"）；同平台但螺丝数不同的计数干扰；用 OAK2 遮挡帧问只有扫描仪能看清的属性。
- 每个任务 5–10 个人工编写的问法模板；评测时要求 JSON 作答。
- 规范格式为统一 JSONL（含物化的合法动作集合与 `graph_version`），再转换为 Qwen-VL / InternVL / LLaVA 格式。
- **测试集**：模板生成 + 100% 人工审核（抽样平衡约 2–3k 条），不做 LLM 改写。
- **评测**：P0 零样本（主流闭源 + Qwen3-VL、InternVL 系列）+ 纯文本基线；P1 人类基线与 LoRA 微调一个开源 VLM；P2 LLM 改写问法（仅训练集）。

---

## 9. 质量控制

### 9.1 自动检查（S7 必须全部通过）

- 每个动作的目标存在；状态变化合法。
- 每个需要掩码的实例在 gold 视角有生效形状，或 `hidden`。
- 同一帧可见掩码两两不重叠。
- 冲突、需复核、未解释变化队列清零。
- 约束图无环；观察序列合法；失败尝试有未满足约束。
- dupli 帧与前一帧掩码一致。

### 9.2 人工质检与上手

- **标注指南**：`docs/annotation_guide.md`，含编辑语义、状态定义、常见情况示例（从 D13 截图）。
- **同学上手**：先独立标 D13 扫描仪视角，与用户版本比较，掩码 IoU ≥ 0.9、状态一致率 ≥ 95% 后进入生产标注。
- **双标**（P1）：4–5 台扫描仪视角两人独立完整标注，报告掩码 IoU、状态 κ、约束边 F1。

---

## 10. 工程架构

### 10.1 目录结构

```
D:\DataSet\
  tda\
    core\         # 无 Qt 依赖
      index.py        # 统一索引、异常修复规则
      logs.py         # Drive 记录导入与解析
      taxonomy.py     # 词表、映射、状态与追踪策略
      db.py           # SQLite 模式、访问层、备份
      masks.py        # RLE、形态学、多边形互转、容差对称差
      geometry.py     # 配准、单应、12MP↔对齐图映射
      states.py       # 状态事件、附着子零件、needs_mask(k)
      compiler.py     # 图层编译器（纯函数）
      graph.py        # 约束规则、模板、校验、合法动作
      checks.py       # 自动检查
      export\coco.py, vlm.py, splits.py
    models\
      sam_service.py  # 交互式 SAM（工作线程）
      trainer.py      # YOLO26-seg 训练子进程
      predictor.py    # 预测插件接口
    ui\
      app.py
      canvas\         # QGraphicsView、标签图渲染、工具
      panels\         # 时间轴、任务卡、实例、步骤表、审阅、约束、质检
      commands.py     # Op 日志、撤销/重做
    cli.py            # build-index / import-logs / export-coco / export-vlm / train / check
  configs\          # taxonomy_map.yaml、index_fixes.yaml、graph_rules.yaml、templates\
  annotations\      # tda.sqlite
  cache\
  raw_logs\
  envs\             # 环境说明与验证报告
  tests\
  docs\
```

### 10.2 关键实现要点

- **渲染**：单张 uint16 标签图（可见掩码）+ 调色板；编辑中实例用独立布尔数组；脏矩形重绘；OAK 12MP 用 ROI 裁剪缓存与金字塔。
- **撤销**：`Op` 日志记录源级操作与逆补丁（RLE 差分）。
- **环境**：conda `tda`（Python 3.11）、PyTorch CUDA 12.8、PySide6、opencv-python、pycocotools、sam2（无 CUDA 扩展）、ultralytics。
- **测试**：编译器（锚点解析、`detached` 双链、环状层级、缺关键帧、冲突生成、附着子零件、追踪策略、遮挡层）、索引异常修复（小型夹具）、约束规则与校验（含 socket_host/cable 与虚拟节点）、导出往返均有单元测试；界面做冒烟测试。

---

## 11. 时间线与容量

### 11.1 开发（多 worker 并行）

| 目标日期 | 交付 |
|---|---|
| 9/17–9/18 | 环境验证（含 SAM 3 / DINOv3 / RF-DETR 可用性）；统一索引 + 记录导入 + 词表映射初稿；数据库模式（**含 §3.1 全部必录字段**）与编译器（含单测）；Label Studio 旧标注导入为草稿 |
| 9/19–9/24 | **MVP（扫描仪）**：画布、画笔/橡皮、SAM 点/框/局部修正、步骤表、倒序任务卡、逐帧确认与审阅、`visibility` 快捷键、备份；约束第一轮用 YAML + CLI 校验 |
| 9/24–9/26 | 用户用 D13 试标，修正编辑语义与快捷键；标注指南初版 |
| 9/26 起 | **开始正式标注**（种子集：4 大族各 2–3 台 + 4 台杂牌，共 12 台，扫描仪 gold） |
| 9/30 前 | **端到端演练**：D13 标注 → COCO → VLM JSONL（≥3 个任务）→ 一次零样本评测，验证 schema 无缺口 |
| 9/26–10/3 | 多边形顶点编辑、差分自动匹配；约束面板；SAM 2.1 / SAM 3 交互效果对比（以 Label Studio 旧标注为参考） |
| 10/3–10/9 | 用种子集训练 YOLO26-seg（并与 DINOv3 骨干分割模型、RF-DETR-Seg 小规模对比）；在第 13 台上量化辅助收益；**10/9 检查点** |
| 10/9–10/17 | 语义角点、跨视角单应、OAK1/OAK2 迁移流程（含堆放区 ROI）、RealSense bronze 流程 |
| 10/17–10/24 | VLM 生成器（P0 任务）与零样本评测脚本；COCO 导出完善 |
| 10/24–11/3 | 自动检查完善、导出与划分、基线准备；P1 VLM 任务 |

### 11.2 标注容量（每天约 6 h，按 6 天/周 ≈ 36 h/周）

- 9/26–11/3 约 5.5 周 ≈ 200 h。
- 需求：种子集 12 台 × 2.5 h + S1/S6 ≈ 40 h；剩余 54 台在模型辅助下 1.2–1.5 h/台 ≈ 65–80 h（若辅助无效则 110–135 h）；S1/S6 其余 ≈ 30 h；OAK1 gold 0.8 h × 66 ≈ 50 h；OAK2 silver 0.5 h × 66 ≈ 33 h；RealSense bronze 0.25 h × 66 ≈ 17 h。合计约 235–290 h。
- **结论**：扫描仪 gold + 约束图在模型辅助有效的前提下可在 11/3 前完成；其余三视角按"OAK1 → OAK2 → RealSense"顺序推进，未核验部分以 `auto` 状态导出并如实标注质量等级。
- **10/9 检查点**（种子集 + 首次训练后）：若辅助后单台扫描仪耗时 ≤ 1.5 h → 维持 66 台 gold；否则切换 R6 的 45 + 21 方案，或将冻结日推至 11/6。

### 11.3 冻结后

- 11/3 数据冻结与划分 → 11/3–11/16 基线、P1 微调、写作（论文骨架 10 月并行起草）。

---

## 12. 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| 5090/Windows 上 SAM 或 PyTorch 不可用 | 阻塞开发 | 第一天验证；备选 WSL2 或 Ultralytics 封装的 SAM |
| 图层编辑语义复杂 | 标注慢、出错 | 默认规则 + "影响 N 帧"提示 + 审阅模式；D13 试标后调整；标注指南 |
| 不脱落螺丝在 OAK 上差分不可见 | 身份/状态误判 | 身份与状态只在扫描仪判定 |
| 扫描仪缺帧 | 主视角缺失 | 逻辑步照常；其他视角回退原生标注 |
| 冲突风暴（小目标重采样误差） | 审阅负担 | 容差对称差 + 面积/像素双阈值 |
| 其他视角域差导致预标质量低 | 核验变慢 | 每视角冷启动先人工 3–4 台 |
| 四视角容量不足 | 部分视角只有 auto 初稿 | 如实标注质量等级；优先核验测试族 |
| 开发延期 | 压缩标注时间 | MVP 仅扫描仪、功能边标边加、多 worker 并行 |
| 约束图缺边 | VLM 答案错误 | 观察序列 / 失败尝试双重校验；测试集人工审核 |
| 数据损坏 | 返工 | WAL + 每日备份 + 操作日志 |

---

## 13. 被否决的方案

| 方案 | 否决原因 |
|---|---|
| 继续使用 Label Studio / CVAT / X-AnyLabeling | 按单图设计，无法表达跨步继承、图层、共享实例、状态与约束 |
| 本地网页应用 | 开发量更大，两人轮流使用时收益不足 |
| 每帧独立掩码 | 遮挡需逐帧人工处理，无近似完整形状与自动遮挡关系 |
| 以图层模型本身作为真值 | 修改会静默改变已确认帧 |
| 关键帧使用显式 [a, b] 区间 | 倒序时下界未知，易产生缝隙 |
| 链式相邻配准 | 漂移累积会静默污染整段 |
| 按接口"端点"生成约束 | 会把 SATA 线两端都变成约束 |
| 追踪拔下的接口 | 插头随线缆移动，代价极高而收益低 |
| 桌面零件另设类别 | 检测语义应保持"同类物体"；`detached` 作属性 |
| 52 类细粒度词表 | 样本不足；状态应为属性 |
| 以实际拆解顺序作为约束真值 | 只是合法顺序之一 |
| 预先划分测试集 | 用户决定先全标；划分在冻结时按平台确定 |

---

## 14. 修订记录

- **v1.6（2026-09-20）**：Plan B 第一、二批的裁定。(1) §7.2 每类边挡住哪些动作的表、`blocked_by` 按方式收窄、被拆走的阻挡物满足任何边、`recommended` = 偏好、规则边的接受/拒绝/孤儿、`graph_version` 含 `status`、Apply 时重推规则边。(2) §7.4 "无环"改为"无动作死锁"（与/或依赖、清除动作是一组候选含"拆走"）、死胡同必须可见。(3) §8 确定真值原则、四种真值来源、V4/V5/V6/V16 的标签形态与指标、排除条件、prompt/label 分离与不透明 id、题面捷径上界表。(4) 实现期其他裁定（不改正文，记录在案）：位姿断点 `pose_break`（库 v4）及重切语义（一次合并只认一个保留方、实测位移 < 25 px 才复制 ROI、已确认帧只进复检队列）；ROI 建议 = 位姿段首/中/尾三帧（扫描仪取并集，OAK 取逐边中位数 + 12 % 外扩且不超面积上限），在后台线程测量；12 MP 帧的编译按形状窗口进行（存储仍为整幅原生坐标，输入哈希不变），编译器发布的掩码只读；叠加层按失效区域、按视口合成；旧 Label Studio 草稿只作可采纳的初稿（`Shift+A` 按光标取最近者，来源 `ls_adopted` 属于一次编辑，记录重叠像素）；本视角差分的斑块拆分实测无显著收益（< 40 px 的零件占 54 % 的事件，候选集本身不含该零件），未接入；指南篇幅按字符计（≤ 7,300）。
- **v1.5（2026-09-19）**：用户裁定 + 两个只读实测。(1) **四个视角各自从头标，视角内一致**；不做标注迁移，4 角点单应 / AprilTag 外参 / 深度迁移只可作可选粗提示，任何流程与排期不得依赖（§1 原则 3、§2.5、§3.2、§4.1 S3–S5、§5.2）。S3–S5 的旧耗时估计作废，待试标实测——**排期风险**：若 OAK1/OAK2 与扫描仪同量级，66 台 × 3 个精标视角超出 11/16 前的人工预算，需在 10/9 检查点按实测决定 OAK2 是否降级或只标子集。(2) **位姿段按视角独立**，新增 `pose_break`（§2.5），审计结果作为提议导入。(3) 主板上的卡扣随主板离开：分类表 `host_class`，`ram_latch`/`cpu_socket_lever` 增加只由级联到达的 `removed` 状态。(4) 确认帧时交回的编译结果必须在真值服务边界用输入哈希证明新鲜。
- **v1.4（2026-09-18）**：实现期裁定汇总。(1) **任务卡属于屏幕上的这一帧**：第 j 帧的任务卡 = 状态(j+1) → 状态(j) 的变化，形状画在当前图像上；差分图与 `Tab` 对比的是刚完成的第 j+1 帧；起点帧列出该帧所有需要形状的实例。(2) **真值表延迟物化**：未验证帧的编译结果只是缓存，提交只同步重算当前帧；已验证帧由后台线程复查（待复查清单持久化，帧状态 `recheck`），导出与检查前先清空待复查并刷新；已验证帧仍只能经冲突队列改变。(3) 交互分割保留 SAM 2.1（与 SAM 3 打平）；默认提示 = 点 + 差分斑块框，单点给 3 个候选（`C` 轮换）；差分图用 ±1 px 最小色差、绝对 ΔE 阈值 12、邻近斑块合并，"未解释变化"判定含包含规则。(4) 快捷键：视角 `F1`–`F4`，可见性 `1`–`7`；有未提交修改时禁止切帧与切实例（状态栏提示，不弹窗）。(5) 状态事件一律由动作推导，人工事件叠加其上；电源线（`atx_24pin`/`cpu_power`/`sata_power`/`molex`）归属电源，`sata_data` 无归属；`remove` 对已插接口合法（计划先 `disconnect`）。(6) 缩略图离线生成，按姿态段 ROI 裁剪。
- **v1.0（2026-09-17）**：整合用户决定与 Fable 第一轮评审。
- **v1.1（2026-09-17）**：整合 Fable 第二轮评审（在场≠需要掩码、线缆虚拟节点、socket_host/cable_owner、锚点关键帧、FrameOverride 统一、容差冲突、23 类、MVP 收缩等）。
- **v1.3（2026-09-17）**：整合 Opus 讨论并由 Fable 裁定——**采纳**：`placement` 作为物理位置属性（不新增状态值）、堆放区只标框 + `bench_annotated`、RealSense 降为 bronze（帧级验收）、`visibility` 枚举替代 hidden/too_small、`mounted_on` 与 `screw.fastens` 拆分 parent 语义、`necessity`/`blocked_by.mode`/`group_order`/`slot_id`/`chassis_frame_in_image`/`occluder_type`/`head_source` 等必录字段、VLM 任务重排（V4/V10/V12/V14/V15/V16 进 P0，V7 降 P1，V13 改为反事实 + 案例）、推理链封闭操作集与落地准确率指标、困难负样本清单、导出物化合法动作集合与 `graph_version`、种子集 12 台 + 10/9 检查点 + 9/30 端到端演练。**未采纳**：RealSense 用深度分层映射（无外参，改为单应提示 + 模型）、堆放区推迟到 P1（用户要求标，改为 P0 只标框）、把 45+21 作为默认计划（用户要求全标，作为检查点后的备选）、约束大量使用 recommended（用户定义为物理必要，默认 required）。
- **v1.2（2026-09-17）**：按用户对 Q1–Q10 的回答修订——RealSense 加回为第 4 视角（silver、`too_small`）；扫描仪固定用 `P_0`；桌面已拆零件以 `placement=detached` 关键帧链建模，类别不变；`Connector.cable` 替代 `cable_owner`，新增 `cable_clip.holds`、`removal_direction`、`failure_reason`、`difficulty`、帧级 `in_progress`；编译器输出遮挡比例；关系章节改为"显式记录 vs 推导"清单；VLM 任务扩展到 V1–V18 并给出设计原则与推理链格式；取消预先划分，划分在冻结时确定；时间线改为多 worker 并行、9/26 开始标注；容量按每天 6 h 重算。
