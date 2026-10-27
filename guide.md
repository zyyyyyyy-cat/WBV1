# B-B V1.0：基于 UAD 的 Query 风险感知与负身份分层排序学习——代码修改实施指导

**研究任务：** WHU-MARS 多模态、多平台行人重识别  
**基础模型：** UAD（Unified Alignment and Discrimination）  
**代码基准：** WHU
**方案版本：** B-B V1.0  
**文档用途：** 指导代码实现、检查修改正确性、执行首次正式训练及后续复现  
**实施原则：** 不增加推理分支，不改变 UAD 主干，不重复已有监督，只替换 WRT 所负责的局部度量学习

---

# 一、方案总目标与实施边界

## 1.1 总体目标

本方案针对 WHU-MARS 的 RGB、NIR、TIR 多模态以及 Ground、Aerial 多平台行人重识别任务，在原 UAD 有效身份表征的基础上，改进训练阶段的图像级身份排序监督。

最终目标是在以下数据集或评测设置中，争取 mAP、Rank-1、Rank-5、Rank-10 和 mINP 较相应 UAD 基线共同提高，并以至少 1.0 个百分点作为主要性能提升目标：

- WHU-MARS-1000；
- WHU-MARS-1000-GD；
- WHU-MARS-2337。

方案不仅关注总体指标，还需观察九种有向模态检索组合、地面与航拍跨平台检索，以及 Rank-1 新成功、新失败的 Query。

**核心思想：** 不再简单聚合出一个加权正样本距离和一个加权负样本距离，而是根据 Query 当前的顶部排序风险、有效正样本覆盖需求，以及真实负身份的混淆程度，将监督预算分配给具体正负图像关系，并通过统一 Pairwise Ranking Loss 更新共享 CLS 表征。

## 1.2 最终改动范围

保留原 UAD：

- 单分支 ViT-B/16；
- 768 维 BN 前 CLS；
- BNNeck 与身份分类器；
- CE；
- ProCA；
- GPD 与 PrototypeBank；
- PKM 数据采样基本流程；
- 原训练优化器、学习率调度和梯度裁剪；
- 推理网络和评测协议。

新增训练期功能：

1. B-B 排序损失；
2. 根据 PID、Camera ID 生成有效候选集合；
3. Query 顶部风险计算；
4. 正样本预算分配；
5. 负身份风险和身份内部困难图像预算分配；
6. 新损失的 Query 级归一化与 DDP 归一化；
7. 必要的数值与梯度诊断。

**不新增可训练网络参数。** 所有新增参数均为损失超参数或训练时动态计算的预算权重。

## 1.3 完整训练目标

原 UAD：

\[
L_{\mathrm{UAD}}=
L_{\mathrm{CE}}+
L_{\mathrm{WRT}}+
0.01L_{\mathrm{ProCA}}+
L_{\mathrm{GPD}}
\]

B-B V1.0：

\[
\boxed{
L_{\mathrm{total}}=
L_{\mathrm{CE}}+
\lambda_B L_B+
0.01L_{\mathrm{ProCA}}+
L_{\mathrm{GPD}}
}
\]

其中：

- B-B **替换** WRT，而不是叠加在 WRT 上；
- CE、ProCA、GPD 保留原权重和原计算方式；
- \(\lambda_B\) 在首次训练前依据真实特征梯度完成一次性标定；
- 首轮不增加独立 Top Loss、Coverage Loss 或困难身份损失。

---

# 二、论文依据、代码依据与本方案创新边界

这一部分是后续论文撰写、代码注释和实验归因的依据。

## 2.1 参考研究与具体位置

| 参考来源 | 相关位置 | 可以借鉴或保留的内容 | 在本方案中的性质 |
|---|---|---|---|
| Zhao 等，*WHU-MARS: A Multispectral Aerial-Ground Benchmark Towards Any-Scenario Person Re-Identification*，CVPR 2026 | §4.1–4.2、式 (1)–(3) | AS-ReID 定义、统一 ViT、PKM、BNNeck、CE+WRT、跨相机评测 | **原论文与基础代码** |
| 同上 | §4.3、式 (4)–(6) | ProCA 两级中心聚合及 stop-gradient 身份中心 | **完整保留原机制** |
| 同上 | §4.4、式 (7)–(9) | GPD 全局身份原型对比与 EMA 原型更新 | **完整保留原机制** |
| 同上 | §4.5、式 (10) | CE、WRT、ProCA、GPD 的联合监督 | **作为总损失修改起点** |
| AGW / WRT，Ye 等，TPAMI 2021 | Weighted Regularization Triplet 机制 | 困难正负距离的自适应加权及局部度量学习 | **B-A 基准来源；B-B 的替换对象** |
| Ramzi 等，*Optimization of Rank Losses for Image Retrieval*，TPAMI 2025 | 可微排名、AP/Recall@k 优化及排序不可分解性 | 从正确匹配排名出发考虑监督，以及 batch 排序与完整检索的差距 | **研究启发；不直接复制 SupRank Loss** |
| MINER，2026 | 多正样本、困难身份选择与身份内外变化协调 | 困难关系平滑选择及多正样本监督的权衡 | **机制启发；不复制其采样器** |



## 2.2 哪些是我们的新增设计

以下内容不是 UAD 论文的原始实现，也不是对 SupRank、MINER 代码的直接移植：

1. 基于 BN 前归一化 CLS 的 Query 级真实图像 Pairwise Ranking Loss；
2. 以最佳正样本与最危险负样本的相似度差计算顶部风险；
3. 根据顶部风险动态调整最佳正样本的额外预算；
4. 使用当前负身份内部最大图像相似度定义身份混淆风险；
5. 在负身份之间、身份内部图像之间分层分配受控预算；
6. 预算权重停止梯度、排序相似度保留梯度的统一监督方式；
7. 使用该损失替换 WRT，并与原 CE、ProCA、GPD 协同训练。

这些共同构成 B-B V1.0 的方法机制。

论文中应准确描述为**受到排序学习与困难关系建模启发，在 UAD 上设计的新的局部排序监督**，而不能声称原论文已经提出这些公式。

## 2.3 论文描述与仓库代码存在差异时的处理规则

实施中按以下优先级处理：

1. **决定代码如何修改：** 以当前 WHU 复现代码及已经成功运行的有效配置为准。
2. **解释原方法的设计意图：** 以 UAD 论文的公式和正文为准。
3. **存在论文—代码差异：** 保持实际复现行为，记录差异，不擅自替换成论文的另一种表述。
4. **本研究的新设计：** 以本指导文档规定的 B-B 公式为准。

已确认的重要对应关系：

| 事项 | 论文 | 当前 WHU 实现 | 本次处理 |
|---|---|---|---|
| Progressive Center Alignment | ProCA | 函数 `PCA()` | 不改函数定义，仅在文档中统一称 ProCA |
| 模态中心 | 归一化样本均值再归一化 | 与论文一致 | 保留 |
| 身份中心 | 停止梯度 | `g_cent.detach()` | 保留 |
| GPD 动量 | 旧原型系数 \(\alpha=0.8\) | `m=0.2` 是新中心系数 | 保留，二者等价 |
| WRT | 原始 BN 前特征的加权 Triplet | 未归一化欧氏距离 | 原 UAD/B-A 保留；B-B 替换 |
| BNNeck | 分类与检索特征分开 | CE 使用 BN 后，推理使用 BN 前 | 保留 |
| 图像增强 | 论文泛称 random resized cropping 等 | 代码实际 Resize→Pad→RandomCrop | 保留当前代码 |
| GPD 更新 | 每 iteration 更新原型 | 优化器 step 后无梯度 EMA 更新 | 保留 |
| DDP 学习率 | 论文描述 0.008 per GPU | 四卡日志生效 `BASE_LR=0.032` | 沿用实际四卡配置 |
| mINP | 主论文主要在 VI-ReID 报告 | 当前评测代码可计算更多设置的 mINP | 保留当前统一评测口径，并与对应复现基线比较 |

另外，论文所说不依赖专门的场景成对对齐机制，不代表基础 WRT 没有正负图像对或三元组计算。不要把这两种概念混淆。

---

# 三、当前 WHU 代码的真实结构与接入位置

## 3.1 核心调用链

当前仓库的训练主线为：

```text
train.py
  ├── 读取 configs/UAD.yml 和命令行覆盖配置
  ├── 初始化 DDP / 随机种子
  ├── make_dataloader()
  │     ├── WHU_MARS 数据解析
  │     ├── PKMSampler_DDP
  │     └── train_collate_fn
  ├── make_model()
  │     └── build_transformer / ViT-B/16
  ├── make_loss()
  │     └── CE + WRT
  ├── make_optimizer() / create_scheduler()
  └── do_train()
        ├── model 前向
        ├── loss_fn：CE + WRT
        ├── PCA：ProCA
        ├── GPD
        ├── loss.backward()
        ├── gradient clipping
        ├── optimizer.step()
        ├── PrototypeBank.update()
        └── validation / checkpoint
```

B-B 只需要替换其中的 WRT 路径，并向新的排序损失提供 camera ID。

ProCA、GPD 不需要改造成新的接口或共同使用 B-B 的预算。

## 3.2 各文件的修改职责

| 文件 | 修改类别 | 主要要求 |
|---|---|---|
| `loss/bb_ranking_loss.py`（建议新增） | **主要新增** | 完整实现候选过滤、顶部风险、正负预算、Pairwise 损失 |
| `loss/make_loss.py` | 必须修改 | 注册 B-B、选择 WRT/B-A/B-B，正确传递 camera ID，确保只计算一项 Metric Loss |
| `processor/processor.py` | 必须修改 | 构造完整模态顺序的 PID/camera，调用 B-B，记录日志，处理 DDP 有效 Query 归一化 |
| `config/defaults.py` | 必须修改 | 加入 B-B 开关、预算和温度参数、\(\lambda_B\) |
| `configs/` | 必须新增实验配置 | 分离原 UAD、B-A、B-B0、B-B1 |
| `loss/wrt_loss.py` | B-A 对照所需 | 提供独立的稳定 WRT 变体，保留原实现可调用 |
| 训练日志/诊断位置 | 少量修改 | 输出必要的候选覆盖、预算、风险与损失信息 |

不需要修改：

- `model/make_model.py` 中的结构；
- `model/backbones/vit_pytorch.py`；
- `datasets/whu_mars.py` 的基本解析；
- `datasets/sampler_ddp.py` 的采样策略；
- `datasets/bases.py`；
- `utils/metrics.py` 的正式评测规则。

可以根据实际工程结构调整新文件名，但功能边界应当保持清晰。

---

# 四、数据流起点：PKM、Query 定义与标签展开

## 4.1 B-B 的 Query 是什么？

**B-B 中的 Query 不是测试集 query，也不是另外采样或初始化的可学习 Query Token。**

训练时，当前 GPU 中的每一张实际图像，都依次作为一个训练 Query。

假设当前 GPU 经过三模态拼接后共有：

\[
N=192
\]

张图像。

则初始候选 Query 索引为：

\[
Q_{\mathrm{all}}=\{0,1,\ldots,N-1\}
\]

这里的“初始化”仅指当前 batch 中的图像索引集合。

不引入：

- Query Embedding；
- 可学习 Query 向量；
- Query 原型；
- Query 特征队列；
- 额外的 Query 初始化参数。

如果某张图像没有有效正样本或负样本，则不参与 B-B 的 Query 平均，但它仍然可能出现在其他 Query 的候选集合中，并继续参与 CE、ProCA、GPD。

## 4.2 PKM 实际采样结构

当前四卡配置：

\[
\mathrm{IMS\_PER\_BATCH}=256,\quad K=4,\quad M=3
\]

每卡获得：

\[
256/4=64
\]

个多模态采样单元。

每卡身份数：

\[
P=64/4=16
\]

因此每身份有：

\[
K\times M=12
\]

张实际图像。

每卡真实模型输入：

\[
16\times12=192
\]

张图像。

训练身份在当前 GPU 的 PKM batch 中按组组织，但 B-B 不应依赖“每四张一定是同一身份”的隐式位置规则来判断 PID，而应直接使用展开后的真实标签。

这样即使后续采样器发生小规模调整，B-B 也不容易因为位置假设而出错。

## 4.3 当前数据加载器的输出

`train_collate_fn` 返回：

- `imgs`：包含 RGB、IR、Thermal 的三个图像 Tensor；
- `pids`：一个多模态采样单元对应一个 PID；
- `camids`：包含三个模态各自 Camera ID 的列表。

若每模态 64 张，则：

```text
imgs[0]  : [64, 3, 256, 128]
imgs[1]  : [64, 3, 256, 128]
imgs[2]  : [64, 3, 256, 128]

pids     : [64]

camids[0]: [64]
camids[1]: [64]
camids[2]: [64]
```

## 4.4 模型中的拼接顺序

原 `build_transformer.forward()` 使用：

```python
x = torch.cat(list(x), dim=0)
```

因此：

\[
X=
[X_{\mathrm{RGB}};
X_{\mathrm{IR}};
X_{\mathrm{Thermal}}]
\]

是**沿 batch 维拼接**。

不是把不同模态图像拼成一个新的多通道输入，也不是把各模态 CLS 拼成一个更高维特征。

拼接后：

\[
X\in\mathbb R^{192\times3\times256\times128}
\]

## 4.5 B-B 标签展开顺序

必须与模型输入严格一致：

\[
Y=
[target;target;target]
\]

对应实现语义：

```python
target_rep = target.repeat(num_modalities)
camids_rep = torch.cat(camids, dim=0)
```

因此：

```text
global_feat : [192, 768]
target_rep  : [192]
camids_rep  : [192]
```

**重要：** 不能直接使用 `target.repeat_interleave(3)`，除非同时改变了模型的实际特征排列。

当前代码不需要那样修改。

B-B 使用的 PID、camera 必须与同一行 CLS 对应同一张图像。

## 4.6 Camera ID 的实际来源

当前 `datasets/whu_mars.py` 从文件名中解析：

```text
PID 与 c(camera_id)
```

然后将相机编号减一。

B-B 直接使用已有 camera ID，不重新构造相机编号。

平台 Ground/Aerial 不能简单替代 camera ID，因为一个平台可能包含多个相机。

同样，不应只根据 RGB/NIR/TIR 模态判断是否为跨相机。

B-B 的 camera 过滤依据必须与正式评测器一致。

---

# 五、前向特征提取：CLS、BNNeck 与检索特征

## 5.1 共享 ViT 前向

三模态输入经过同一套 ViT-B/16 参数：

\[
f_i=f_\theta(x_i)
\]

输出：

\[
F_{\mathrm{raw}}\in\mathbb R^{192\times768}
\]

即当前代码中的：

```python
global_feat
```

此特征是最后一层 ViT 的 CLS 表征。

不需要：

- 中间层 Token；
- Patch Token Pooling；
- 新建 512 维局部特征；
- 对 RGB/NIR/TIR 分别建立输出头。

## 5.2 CE 的特征路径

原代码执行：

```python
feat = self.bottleneck(global_feat)
cls_score = self.classifier(feat)
```

因此：

\[
F_{\mathrm{BN}}=\mathrm{BNNeck}(F_{\mathrm{raw}})
\]

CE 使用：

\[
L_{\mathrm{CE}}
=
\mathrm{CE}(W_{\mathrm{cls}}F_{\mathrm{BN}},Y)
\]

原 `IF_LABELSMOOTH='off'` 时使用普通交叉熵。

保持不变。

## 5.3 原 WRT 的特征路径

原 WRT 使用：

\[
F_{\mathrm{raw}}
\]

构造未归一化欧氏距离。

这属于原实现行为。

B-B 替换该局部度量损失后，不需要保留一条 WRT 距离路径参与总损失。

## 5.4 B-B 的特征来源

B-B 必须使用：

\[
\boxed{F_{\mathrm{raw}}=\texttt{global\_feat}}
\]

随后在新损失内部执行：

\[
\boxed{
Z=\mathrm{L2Norm}(F_{\mathrm{raw}})
}
\]

不允许误用：

- `feat`；
- `cls_score`；
- 分类器权重；
- GPD 的 prototype；
- ProCA 的身份中心；
- 测试评测器缓存的特征。

## 5.5 与推理特征的对应关系

原推理设置：

```yaml
TEST:
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'
  RE_RANKING: False
```

模型返回 BN 前 CLS，由评测器执行 L2 归一化。

所以：

\[
z_i=\frac{f_i}{\|f_i\|_2}
\]

与 B-B 的训练监督空间一致。

对于单位归一化特征：

\[
\|z_q-z_g\|_2^2=2-2z_q^\top z_g
\]

因此余弦相似度越高，对应的归一化欧氏距离越小。

B-B 不改变实际测试的特征维度与排序方向。

---

# 六、B-B 的 FP32 特征归一化与相似度计算

## 6.1 操作顺序

B-B 必须按以下顺序执行：

```text
BN 前 CLS
   ↓
转为 FP32
   ↓
沿特征维执行 L2 Normalize
   ↓
计算余弦相似度矩阵
   ↓
构造候选掩码
   ↓
计算风险与预算
   ↓
Pairwise Ranking Loss
```

不应将候选掩码作用到 CLS 特征上再归一化。

也不应先计算未归一化的欧氏距离，再在距离矩阵上做简单标准化作为替代。

## 6.2 FP32 转换

对当前：

\[
F_{\mathrm{raw}}
\]

使用：

\[
F_{32}=F_{\mathrm{raw}}.\mathrm{float}()
\]

推荐将 B-B 核心数值计算置于禁用 autocast 的局部上下文中。

原模型前向、CE、ProCA、GPD 仍按照现有 AMP 设置运行。

无需将整个 UAD 改为 FP32 训练。

## 6.3 L2 归一化

定义：

\[
\boxed{
z_i=
\frac{f_i}{\max(\|f_i\|_2,\epsilon)}
}
\]

首轮：

\[
\epsilon=10^{-12}
\]

得到：

\[
Z\in\mathbb R^{192\times768}
\]

推荐采用框架提供的稳定 `F.normalize`。

如果特征正常，不应因为该 epsilon 改变模型几何。

## 6.4 计算相似度矩阵

\[
\boxed{S=ZZ^\top}
\]

形状：

\[
S\in\mathbb R^{192\times192}
\]

第 \(q\) 行为当前 Query 与所有本地图像的相似度。

\[
S_{qn}=z_q^\top z_n
\]

**这一矩阵必须保留计算图和梯度。**

不要直接计算：

```python
S = (Z @ Z.T).detach()
```

否则 B-B 无法更新共享 ViT。

## 6.5 用于预算的相似度副本

将预算计算与实际损失计算分开：

```python
S = Z @ Z.T
S_budget = S.detach()
```

或者在预算计算部分使用 `torch.no_grad()`。

二者的职责不同：

- `S_budget`：计算 Top、身份风险与动态预算；
- `S`：计算真正的正负排序损失并进行反向传播。

预算矩阵不应替代损失中的相似度矩阵。

---

# 七、有效候选集合与 Query 筛选

## 7.1 为什么要做相机过滤？

WHU 当前评测器对每个 Query 执行：

```python
remove = (g_camids[order] == q_camid)
```

即移除与 Query 相同 Camera ID 的所有 Gallery 候选。

这一规则不仅针对同 PID、同相机正样本，同相机的负身份候选也会被移除。

B-B V1.0 采用一致的跨相机关系定义。

这里需要清楚：这是 **B-B 新增的训练候选规则**，原 WRT 并未执行完全相同的相机过滤。

## 7.2 正样本集合

对于训练 Query \(q\)：

\[
\boxed{
P_q=
\{p:
y_p=y_q,\ 
p\ne q,\ 
c_p\ne c_q
\}
}
\]

必须同时满足：

1. PID 相同；
2. 不是 Query 自身；
3. Camera ID 不同。

不得使用所有 PID 相同的样本作为有效正样本。

## 7.3 负样本集合

\[
\boxed{
N_q=
\{n:
y_n\ne y_q,\ 
c_n\ne c_q
\}
}
\]

必须同时满足：

1. PID 不同；
2. Camera ID 不同。

正负集合依据当前本地 batch 构造。

## 7.4 Query 自身为什么必须排除？

归一化后的：

\[
s(q,q)\approx1
\]

如果把 Query 自己当作正样本，最佳正样本就会被自己的相似度占据。

这样：

\[
\Delta_q
\]

不再能真实反映其他同身份图像是否超过最危险负样本。

这会直接破坏后续的风险预算。

因此，自身排除是硬性要求，不是可选实现。

## 7.5 有效 Query

定义：

\[
\mathcal Q_{\mathrm{valid}}
=
\{q:\ |P_q|>0,\ |N_q|>0\}
\]

若缺少任一集合：

- 不进行 Max；
- 不执行该 Query 的 Softmax；
- 不计算该 Query 的 B-B Loss；
- 不计入有效 Query 数量。

但该图像仍可能作为其他 Query 的候选，并且仍参加原 CE、ProCA、GPD 训练。

## 7.6 特殊情况

当有效正样本仅有一个时：

\[
|P_q|=1
\]

对应的正样本预算必定等于 1。

此时 Top 预算机制不会产生实际的正样本之间分配差异。

如果某 GPU 上没有有效 Query，不应直接制造 NaN 或除零。

对于空集合，必须采用明确且一致的跳过规则。

## 7.7 本版不引入专门跨模态过滤

虽然 WHU 的困难包含跨模态差异，但 B-B 不要求：

\[
m_p\ne m_q
\]

作为有效正样本条件。

原因是最终 Mixed Gallery 同时包含：

- 同模态跨相机；
- 跨模态跨相机；
- 不同平台与场景组合。

使用统一有效跨相机关系，更符合 AS-ReID 的总体检索目标。

也不额外按 Ground/Aerial 设置手工固定损失权重。

---

# 八、Query 顶部风险及正样本预算

## 8.1 最佳有效正样本

在当前 Query 有效正集合中：

\[
p_q^*=
\arg\max_{p\in P_q}s_{qp}
\]

得到：

\[
s_q^+=s_{qp_q^*}
\]

## 8.2 最危险有效负样本

在有效负集合中：

\[
n_q^*=
\arg\max_{n\in N_q}s_{qn}
\]

得到：

\[
s_q^-=s_{qn_q^*}
\]

## 8.3 顶部间隔

定义：

\[
\boxed{
\Delta_q=s_q^+-s_q^-
}
\]

解释：

- \(\Delta_q<0\)：当前 batch 中最危险负样本超过最佳正样本；
- \(\Delta_q=0\)：正负顶部处于交换边界；
- \(\Delta_q>0\)：最佳正样本暂时领先。

这里判断的是当前训练候选，不是完整测试 Gallery 上的真实 Rank-1。

## 8.4 Query 风险

定义：

\[
\boxed{
r_q=\sigma(-\Delta_q/\tau_r)
}
\]

首轮：

\[
\boxed{\tau_r=0.05}
\]

\(\tau_r\) 负责控制顶部风险随间隔变化的敏感程度。

当顶部错误更严重时，风险趋近 1。

当顶部安全优势增大时，风险趋近 0。

该风险计算不参与反向传播。

## 8.5 顶部额外预算

定义：

\[
\boxed{
\alpha_q=\alpha_{\max}r_q
}
\]

首轮：

\[
\boxed{\alpha_{\max}=0.5}
\]

它是固定超参数，不是网络参数。

每个 step 使用当前特征重新计算 \(\alpha_q\)。

## 8.6 正样本预算

\[
\boxed{
a_{qp}
=
\frac{1-\alpha_q}{|P_q|}
+
\alpha_q\mathbf 1[p=p_q^*]
}
\]

其中：

- 第一项给予全部正样本基础预算；
- 第二项给予最佳正样本风险感知的额外预算。

满足：

\[
\sum_{p\in P_q}a_{qp}=1
\]

并且：

\[
a_{qp}\ge0
\]

## 8.7 设计目的与实际作用

当顶部排序错误时：

- 最佳正样本获得更多预算；
- 其余正样本仍获得非零预算；
- 不会像 WHV2 那样只对少量教师 Focus Query 提供 Top 监督。

当顶部排序较安全时：

- 最佳正样本的额外预算减少；
- 监督更多回到其他有效正样本；
- 有利于继续改善多正样本的整体排名。

这一分配不能保证 Rank-1 永不下降，但能够明确表达我们希望改善的顶部竞争机制。

## 8.8 关于 margin 的职责

顶部风险使用：

\[
\Delta_q=0
\]

作为中心。

Pairwise Loss 的安全 margin：

\[
m=0.05
\]

只在排序损失中使用。

两者分离，避免把“当前是否认错第一名”和“希望保留多少安全间隔”混为一谈。

---

# 九、负身份风险与身份内图像预算

## 9.1 按 PID 分组

对于 Query \(q\)，把其有效负样本按 PID 分为：

\[
N_{q,1},N_{q,2},\ldots,N_{q,J_q}
\]

其中：

- \(J_q\) 是有效负身份数量；
- \(K_{q,j}=|N_{q,j}|\)；
- 每个负身份可以包含不同数量的有效图像。

即使原 PKM 每身份采样数量相同，经过同相机过滤后：

\[
K_{q,j}
\]

也可能不同。

因此，不能在实现中始终假设每个负身份都有 12 张有效图像。

## 9.2 身份困难度

定义：

\[
\boxed{
h_{qj}
=
\max_{n\in N_{q,j}}s_{qn}
}
\]

这里的 Max 用于找出某个错误身份中最危险的真实图像。

不使用该身份图像的简单均值代替，否则可能掩盖只有一张特别危险图像的错误身份。

同时，Max 只用于预算，不直接形成单对 Hardest Triplet。

## 9.3 负身份风险分布

\[
v_{qj}
=
\operatorname{softmax}_{j}
\left(
\frac{h_{qj}}{\tau_w}
\right)
\]

首轮：

\[
\boxed{\tau_w=0.10}
\]

注意必须先收集当前 Query 的有效负身份，再执行身份间 Softmax。

不能把负图像列表直接视为负身份列表。

## 9.4 身份总预算

\[
\boxed{
c_{qj}
=
\frac{1-\beta}{J_q}
+
\beta v_{qj}
}
\]

首轮：

\[
\boxed{\beta=0.5}
\]

满足：

\[
\sum_jc_{qj}=1
\]

该设计既保留身份覆盖，又允许危险负身份获得更大的显式监督预算。

不要求所有负身份严格平分预算。

## 9.5 身份内困难图像分布

对于身份 \(j\) 中的图像：

\[
\pi_{qn|j}
=
\operatorname{softmax}_{n\in N_{q,j}}
\left(
\frac{s_{qn}}{\tau_w}
\right)
\]

只在本身份内部归一化。

不得把所有负图像先一起 Softmax，再错误地将输出解释为身份内权重。

## 9.6 身份内部预算

\[
\boxed{
d_{qn|j}
=
\frac{1-\rho}{K_{q,j}}
+
\rho\pi_{qn|j}
}
\]

首轮：

\[
\boxed{\rho=0.5}
\]

其含义是：

- 一半显式预算为身份内部均匀覆盖；
- 一半根据实际图像相似度向高分图像倾斜。

满足：

\[
\sum_{n\in N_{q,j}}d_{qn|j}=1
\]

## 9.7 最终负样本预算

\[
\boxed{
b_{qn}
=
c_{qj}d_{qn|j}
}
\]

其中 \(n\in N_{q,j}\)。

满足：

\[
\sum_{n\in N_{q,j}}b_{qn}=c_{qj}
\]

以及：

\[
\boxed{
\sum_{n\in N_q}b_{qn}=1
}
\]

## 9.8 为什么不是再增加困难身份损失？

因为：

\[
c_{qj}
\]

只决定负身份在统一排序损失中的权重。

没有额外添加：

\[
L_{\mathrm{HardID}}
\]

也不调用新的身份原型分类器。

B-B 仍然只有一个局部排序损失。

## 9.9 动态权重的停止梯度规则

以下预算计算值均不应向上游传播梯度：

- 顶部 Argmax 选择；
- \(\Delta_q\) 在风险预算中的计算路径；
- \(\alpha_q\)；
- \(a_{qp}\)；
- \(h_{qj}\)；
- \(v_{qj}\)；
- \(c_{qj}\)；
- \(\pi_{qn|j}\)；
- \(d_{qn|j}\)；
- \(b_{qn}\)。

但同一个：

\[
s_{qn}
\]

在 Pairwise Loss 中必须继续保持梯度。

正确原则是：

**预算停止梯度，排序相似度不停止梯度。**

---

# 十、统一 Pairwise Ranking Loss

## 10.1 排序错误变量

对于有效三元关系：

\[
(q,p,n)
\]

定义：

\[
e_{qpn}=s_{qn}-s_{qp}
\]

当：

\[
e_{qpn}>0
\]

说明该错误图像排在该正确图像之前。

## 10.2 安全间隔

首轮固定：

\[
\boxed{m=0.05}
\]

希望模型逐渐满足：

\[
s_{qp}\ge s_{qn}+m
\]

这里的 margin 采用余弦相似度尺度。

不能直接解释为原 WRT 中的欧氏距离 margin。

## 10.3 带温度前置系数的 Softplus

定义：

\[
\boxed{
\ell_{qpn}
=
\tau_\ell
\operatorname{softplus}
\left(
\frac{m+s_{qn}-s_{qp}}{\tau_\ell}
\right)
}
\]

首轮：

\[
\boxed{\tau_\ell=0.05}
\]

其导数：

\[
\frac{\partial\ell_{qpn}}
{\partial(s_{qn}-s_{qp})}
=
\sigma\left(
\frac{m+s_{qn}-s_{qp}}{\tau_\ell}
\right)
\]

因此单对排序差的导数介于 0 和 1。

采用前置温度，是为了使温度主要影响排序梯度的平滑范围，而不额外引入最大 \(1/\tau_\ell\) 梯度放大。

## 10.4 Query 级损失

\[
\boxed{
L_B(q)
=
\sum_{p\in P_q}
\sum_{n\in N_q}
a_{qp}b_{qn}
\tau_\ell
\operatorname{softplus}
\left(
\frac{m+s_{qn}-s_{qp}}{\tau_\ell}
\right)
}
\]

必须理解计算顺序：

1. 获取每个真实正负图像对的相似度；
2. 计算该对的 Softplus 排序损失；
3. 乘对应正样本预算；
4. 乘对应负图像预算；
5. 对所有正负对求和；
6. 得到当前 Query 的标量损失。

**不能先求加权正相似度和加权负相似度，再对它们计算一次 Softplus。**

因为：

\[
\sum_{p,n}a_pb_n\,\operatorname{softplus}(s_n-s_p)
\]

一般不等于：

\[
\operatorname{softplus}
\left(
\sum_nb_ns_n-\sum_pa_ps_p
\right)
\]

前者保留具体正负图像关系，后者又退化为聚合统计量监督。

## 10.5 本地 Query 损失和

设当前 GPU 有：

\[
N_r
\]

个有效 Query。

计算：

\[
S_r
=
\sum_{q\in\mathcal Q_r^{valid}}L_B(q)
\]

注意这里先求和，不一定立即做本地有效 Query 平均。

四卡 DDP 的最终归一化另行处理。

## 10.6 B-B0 与 B-B1

**B-B0：**

\[
a_{qp}=\frac1{|P_q|}
\]

负身份预算、身份内部预算、Pairwise Loss 与 B-B1 完全一致。

**B-B1：**

\[
a_{qp}
=
\frac{1-\alpha_q}{|P_q|}
+
\alpha_q\mathbf1[p=p_q^*]
\]

除此之外不改变任何训练机制。

这样才能单独判断 Query 风险感知正样本预算是否有增益。

---

# 十一、B-B 梯度传播的详细机制

## 11.1 正样本相似度梯度

设：

\[
u_{qpn}
=
\frac{m+s_{qn}-s_{qp}}{\tau_\ell}
\]

由于权重停止梯度：

\[
\boxed{
\frac{\partial L_B(q)}{\partial s_{qp}}
=
-a_{qp}
\sum_n b_{qn}\sigma(u_{qpn})
}
\]

因此：

\[
\frac{\partial L_B(q)}{\partial s_{qp}}\le0
\]

梯度下降对该 Query 的直接监督方向是提高正样本相似度。

## 11.2 负样本相似度梯度

\[
\boxed{
\frac{\partial L_B(q)}{\partial s_{qn}}
=
b_{qn}
\sum_pa_{qp}\sigma(u_{qpn})
}
\]

因此：

\[
\frac{\partial L_B(q)}{\partial s_{qn}}\ge0
\]

梯度下降对该 Query 的直接监督方向是降低负样本相似度。

## 11.3 预算约束与梯度上界

由于：

\[
\sum_pa_{qp}=1
\]

\[
\sum_nb_{qn}=1
\]

得到：

\[
\left|
\frac{\partial L_B(q)}{\partial s_{qp}}
\right|
\le a_{qp}
\]

\[
0\le
\frac{\partial L_B(q)}{\partial s_{qn}}
\le b_{qn}
\]

对负身份 \(j\)：

\[
\boxed{
\sum_{n\in N_{q,j}}
\left|
\frac{\partial L_B(q)}{\partial s_{qn}}
\right|
\le c_{qj}
}
\]

这些上界解释了预算机制如何限制单个 Query 对不同负身份的直接相似度梯度。

但它们不等于网络参数梯度上界。

## 11.4 从相似度到归一化特征

因为：

\[
s_{ij}=z_i^\top z_j
\]

对单个相似度：

\[
\frac{\partial s_{ij}}{\partial z_i}=z_j
\]

\[
\frac{\partial s_{ij}}{\partial z_j}=z_i
\]

所以梯度会流向：

- 当前 Query 图像的特征；
- 当前正样本图像的特征；
- 当前负样本图像的特征。

不是只更新 Query 一侧。

由于所有样本共享同一编码器，一个图像也可能同时作为其他 Query 的候选，因此它接收到的最终梯度是多条关系贡献的和。

## 11.5 从归一化特征到原始 CLS

对于正常非零特征：

\[
z=\frac{f}{\|f\|_2}
\]

其 Jacobian 为：

\[
\frac{\partial z}{\partial f}
=
\frac1{\|f\|_2}
(I-zz^\top)
\]

因此，B-B 的相似度梯度经过 L2 归一化映射，进一步回传到 BN 前 CLS。

这里有两个意义：

第一，B-B 主要针对归一化特征的方向建立排序监督。

第二，原始 CLS 的范数会影响该梯度映射的尺度。

因此，即使相似度层面的导数有界，也不意味着 B-B 与原 WRT 对共享 ViT 的梯度强度自然一致。

这正是需要进行 \(\lambda_B\) 校准的原因。

## 11.6 梯度到共享 ViT

\[
L_B
\rightarrow
S
\rightarrow
Z
\rightarrow
F_{\mathrm{raw}}
\rightarrow
\mathrm{ViT}
\]

更新原共享 ViT 的可训练参数。

**B-B 不直接更新：**

- BNNeck 的参数；
- 身份分类器的参数；
- GPD PrototypeBank；
- 任何新建局部模块，因为本方案没有新建局部模块。

不过，共享 ViT 更新会改变下一次 CE、ProCA 和 GPD 所看到的输入特征。

## 11.7 动态预算为何不参与反向传播？

假设未停止梯度：

\[
L=a\,b\,\ell
\]

则：

\[
\nabla L
=
ab\nabla\ell
+
b\ell\nabla a
+
a\ell\nabla b
\]

后两项会让模型通过改变预算本身来降低损失。

本版只保留：

\[
ab\nabla\ell
\]

因此：

- 风险和困难度决定当前 step 的监督份额；
- Pairwise Loss 决定当前 step 的梯度方向；
- 下一 step 再根据新特征重新计算预算。

不增加可微权重优化器。

---

# 十二、原 CE、ProCA、GPD 的保留方式与反向路径

## 12.1 CE

保持：

\[
L_{\mathrm{CE}}
=
\mathrm{CE}
\left(
W_{\mathrm{cls}}\mathrm{BNNeck}(F_{\mathrm{raw}}),
Y
\right)
\]

梯度路径：

```text
CE
 ↓
分类 logits
 ↓
分类器
 ↓
BNNeck
 ↓
原始 CLS
 ↓
ViT
```

不改变 CE 权重、标签平滑设置或分类器参数。

## 12.2 ProCA

参考 UAD 论文 §4.3、式 (4)–(6)。

代码对应：

```text
processor/processor.py :: PCA()
```

流程：

1. 从 BN 前 CLS 获取三模态特征；
2. 对每张样本进行归一化；
3. 按每身份每模态 K=4 张图像计算模态中心；
4. 对模态中心再次归一化；
5. 聚合三模态中心得到身份中心；
6. 对身份中心归一化；
7. 将身份中心停止梯度；
8. 计算不同模态中心到身份中心的平方距离。

保持：

\[
\lambda_{\mathrm{ProCA}}=0.01
\]

ProCA 直接更新共享 ViT，不更新一个独立中心网络。

B-B 不修改 ProCA 的聚合顺序。

**特别注意：** ProCA 中的中心不是 B-B 负身份预算中的 \(h_{qj}\)。两者来源、含义与梯度作用完全不同。

## 12.3 GPD

参考 UAD 论文 §4.4、式 (7)–(9)。

保留：

- 768 维训练身份原型；
- 当前数据集对应的训练身份类别数；
- GPD 温度 0.03；
- 原动量更新规则；
- 原本的跨卡身份中心同步；
- 原型 buffer 的无梯度更新。

代码的：

```python
GPD_MOMENTUM = 0.2
```

代表新中心混合系数为 0.2，对应论文旧原型系数：

\[
\alpha=0.8
\]

不要误改为代码 `GPD_MOMENTUM=0.8`。

GPD 损失使用当前特征对全部训练身份原型分类式对比，其梯度流向：

```text
GPD CE
 ↓
归一化图像特征 × PrototypeBank
 ↓
原始 CLS
 ↓
共享 ViT
```

PrototypeBank 不通过反向传播更新。

## 12.4 GPD 更新顺序

保留：

```text
读取当前 PrototypeBank
      ↓
计算本 step GPD Loss
      ↓
与 CE / B-B / ProCA 汇总
      ↓
反向传播
      ↓
优化器更新
      ↓
使用本 batch 的 detached 身份中心
更新 PrototypeBank
```

不要让 B-B 的风险预算影响 GPD Bank 的更新幅度。

不要因为 B-B 引入了 PID 分组而复用其身份困难统计量更新全局原型。

## 12.5 共享主干的总梯度

\[
\boxed{
g_\theta
=
g_{\mathrm{CE}}
+
\lambda_Bg_B
+
0.01g_{\mathrm{ProCA}}
+
g_{\mathrm{GPD}}
}
\]

梯度在共享 ViT 参数上汇合，然后由原优化器更新。

不增加 PCGrad、GradNorm、教师梯度保护或第二套优化器。

---

# 十三、训练处理函数的接入方案

## 13.1 原有接口需要扩展

原 `make_loss.py` 的局部损失接口主要接受：

```text
cls_score
global_feat
target_rep
```

但 B-B 还需要：

```text
camids_rep
```

因此，建议扩展局部损失调用接口，而不是为了让 B-B 工作去修改整个模型的 `forward` 返回值。

推荐职责为：

```text
processor:
    准备展开后的 PID 和 camera
    调用 loss_fn

make_loss:
    选择 CE / WRT / B-A / B-B
    控制 Metric Loss 系数

B-B Loss:
    使用 CLS、PID、camera
    完成全部排序预算与 Pairwise 损失
```

## 13.2 模型 `forward` 保持原样

仍然：

```text
model(imgs, target, camids)
    → cls_score
    → global_feat
    → feat
```

不要求模型额外返回：

- 中间层 Token；
- Query 特征；
- 模态融合向量；
- 平台向量；
- 排序权重。

B-B 所需信息由原数据加载器和现有 `global_feat` 构造。

## 13.3 损失选择开关

建议增加清晰的 Metric Loss 类型，例如：

```text
wrt
wrt_fixed
bb
```

其中：

- `wrt`：原始官方 WRT；
- `wrt_fixed`：仅修正权重数值行为的 B-A；
- `bb`：本研究 B-B 排序目标。

再通过一个独立配置项决定：

```text
BB_POS_MODE = uniform / adaptive
```

对应 B-B0 与 B-B1。

不建议为了 B-B 在 `LOSS_TYPE` 中增加一套会意外关闭 ProCA/GPD 的复杂字符串组合。

原：

```text
LOSS_TYPE = base+pca+gpd
```

继续表达 CE、局部度量、ProCA、GPD 联合训练。

## 13.4 确保 \(\lambda_B\) 只乘一次

应明确约定：

- B-B 模块输出原始、未乘 \(\lambda_B\) 的排序损失；
- 在 `make_loss` 或其明确指定的唯一汇总位置施加 \(\lambda_B\)；
- `processor` 汇总 ProCA、GPD 时不重复乘 \(\lambda_B\)。

避免出现：

\[
\lambda_B^2 L_B
\]

或者：

\[
L_{\mathrm{WRT}}+\lambda_B L_B
\]

这种非预期的训练目标。

日志应同时区分：

```text
BB_raw
BB_weighted
CE
ProCA_raw
GPD_raw
Total
```

便于发现重复加权。

---

# 十四、四卡 DDP 下的有效 Query 归一化

## 14.1 为什么需要处理？

每 GPU 虽然有相同数量的 192 张图像，但经过跨相机正负过滤后，各 Rank 的有效 Query 数量可能不同。

若每 GPU 分别做：

\[
L_{B,r}=\frac{S_r}{N_r}
\]

再让 DDP 平均梯度，得到的是每卡平均损失的平均，而不是全局所有有效 Query 的统一平均。

## 14.2 推荐处理

定义：

\[
S_r=
\sum_{q\in\mathcal Q_r^{valid}}L_B(q)
\]

\[
N_r=
|\mathcal Q_r^{valid}|
\]

GPU 数量：

\[
W=4
\]

对 \(N_r\) 执行跨 Rank 求和：

\[
N_{\mathrm{all}}=\sum_rN_r
\]

每 Rank 在 B 分支使用：

\[
\boxed{
L_{B,r}^{DDP}
=
\frac{WS_r}{N_{\mathrm{all}}}
}
\]

由于标准 DDP 对梯度进行平均，最终与全局有效 Query 平均一致。

## 14.3 不需要跨卡特征 Gather

本方案跨卡只同步一个或少量标量计数。

不执行：

- 所有 768 维特征的 AllGather；
- 跨 GPU Pairwise 相似度计算；
- 带梯度的远端候选构造；
- 跨 batch 特征队列。

因此，每个 Query 的有效候选仍然仅来自本地 GPU。

## 14.4 空 Query 情况

若某 Rank 没有有效 Query：

- 该 Rank 的 \(S_r\) 应当为零；
- 仍正常参与全局 \(N_{\mathrm{all}}\) 计算；
- 不访问空集合 Max/Softmax。

若全部 Rank 都无有效 Query：

\[
N_{\mathrm{all}}=0
\]

则该 step 的 B-B 为零，其他 CE、ProCA、GPD 仍正常计算。

实现时应确保零损失与当前计算图兼容，并避免分布式执行分支不一致。

## 14.5 梯度裁剪与 AMP

沿用原：

1. 汇总全部损失；
2. GradScaler 缩放；
3. 统一 backward；
4. unscale；
5. 原全局梯度裁剪；
6. 优化器 step；
7. GradScaler update；
8. 原 PrototypeBank 更新。

不对 B-B 单独进行第二次参数更新。

---

# 十五、参数与系数的完整配置

## 15.1 B-B V1.0 参数

| 参数 | 首轮初值 | 含义 | 梯度属性 |
|---|---:|---|---|
| \(D\) | 768 | CLS 维度 | 固定结构 |
| \(\epsilon\) | \(10^{-12}\) | L2 归一化稳定项 | 固定 |
| \(m\) | **0.05** | 正负排序安全间隔 | 固定 |
| \(\tau_r\) | **0.05** | 顶部风险变化敏感度 | 固定 |
| \(\tau_w\) | **0.10** | 负身份与身份内困难分布温度 | 固定 |
| \(\tau_\ell\) | **0.05** | Pairwise Softplus 平滑温度 | 固定 |
| \(\alpha_{\max}\) | **0.5** | 最佳正样本最大额外预算 | 固定 |
| \(\beta\) | **0.5** | 负身份风险分配比例 | 固定 |
| \(\rho\) | **0.5** | 身份内部困难图像分配比例 | 固定 |
| \(\lambda_B\) | 梯度标定获得 | 新 Metric Loss 强度 | 标定后固定 |
| \(\Delta_q\) | 动态 | Query 顶部间隔 | 预算路径停止梯度 |
| \(a_{qp}\) | 动态 | 正样本预算 | 停止梯度 |
| \(b_{qn}\) | 动态 | 负样本预算 | 停止梯度 |
| \(S_{ij}\) | 动态 | 图像相似度 | 保留梯度 |

以上固定数值是首次实验初值，不是已经由数据验证的最优取值。

## 15.2 原 UAD 参数

按照已成功运行的四卡复现配置：

| 参数 | 保留设置 |
|---|---|
| Backbone | ViT-B/16 |
| Pretrain | ImageNet 预训练 |
| 输入尺寸 | 256×128，按 H×W |
| GPU | 4 |
| 实际每卡输入图像 | 192 |
| 每卡身份 | 16 |
| 每身份每模态实例 | 4 |
| Optimizer | SGD |
| 总 Epoch | 120 |
| 四卡有效 `BASE_LR` | 0.032 |
| Warm-up | 5 Epoch |
| Scheduler | 原 Cosine 日程 |
| CE 权重 | 1 |
| ProCA 权重 | 0.01 |
| GPD 权重 | 1 |
| GPD 温度 | 0.03 |
| GPD 新中心动量 | 0.2 |
| 训练数据增强 | 原 WHU 实现 |
| 推理 CLS | BN 前 768D |
| 测试归一化 | 开启 |
| Re-ranking | 关闭 |

不要因为代码增加了 B-B，就线性放大学习率或 batch size。

## 15.3 \(\lambda_B\) 的一次性标定

### 标定目的

新 B-B 使用归一化余弦空间，原 WRT 使用未归一化欧氏距离。

即使损失值接近，梯度规模也可能差异很大。

因此采用真实 CLS 梯度范数作为标定依据。

### 标定数据

建议取 **12 个真实 PKM batch**。

在与正式训练一致的 ImageNet 初始化、四卡数据分布和输入增强设置下进行。

标定期间：

- 不更新模型参数；
- 不修改优化器状态；
- 不更新 GPD Bank；
- 使用与正式训练一致的 B-B 数学定义；
- 对不同损失分别计算梯度，不将它们混在一起比较。

### 比较对象

计算：

\[
G_W=
\operatorname{median}_b
\left\|
\frac{\partial L_{\mathrm{WRT}}^{(b)}}
{\partial F_{\mathrm{raw}}^{(b)}}
\right\|_F
\]

\[
G_B=
\operatorname{median}_b
\left\|
\frac{\partial L_{B0}^{(b)}}
{\partial F_{\mathrm{raw}}^{(b)}}
\right\|_F
\]

同时记录：

\[
\left\|
\frac{\partial L_{\mathrm{GPD}}}
{\partial F_{\mathrm{raw}}}
\right\|_F
\]

作为原有监督强度参考。

推荐将修正数值后的 WRT 梯度作为规范化的主要尺度参照，同时记录原始 WRT 梯度，以判断修正是否造成明显的强度差异。

如果实际修正后的 WRT 与原官方 WRT 差异显著，标定报告中必须保留两者数值，不能将差异隐藏。

### 初始系数

按照约定的 WRT 参考梯度：

\[
\boxed{
\lambda_B=
\frac{G_W}{G_B+\epsilon_g}
}
\]

标定值不应直接因数值异常而被接受。

如果 \(G_B\) 接近零、有效 Query 大量缺失或梯度非有限，需要先修正实现问题。

在此基础上检查加权 B 梯度与 CE、GPD 梯度是否处于合理的训练量级。

### B-B0 与 B-B1 的公平性

两者使用同一个：

\[
\lambda_B
\]

建议统一依据 B-B0 完成系数标定。

B-B1 的梯度分布因为自适应正样本预算而变化，是该机制本身的作用，不应为了把两者梯度强行调成相同范数而分别重选系数。

正式训练开始后：

**\(\lambda_B\) 固定，不动态更新。**

---

# 十六、训练初始化与正式实验设置

## 16.1 正式初始化

第一版不使用训练完成的 UAD checkpoint 进行主要正式对照。

采用与原 UAD 一致的 ImageNet ViT-B/16 预训练权重。

保持：

- 原模型构造；
- 原分类器初始化；
- 原 BNNeck 初始化；
- 原 ProCA/GPD 状态初始化；
- 原随机种子；
- 原训练数据；
- 原 120 Epoch 日程。

目的是使 B-B 与原 UAD 在相同训练预算下直接比较。

## 16.2 为什么不采用两阶段训练？

本方案没有需要先独立训练的新网络分支。

因此没有必须采用 60+60 阶段切换的结构性理由。

之前 HiHR 的两阶段训练还涉及优化器状态、学习率重新 Warm-up 和新监督介入等因素。

B-B 首轮不重复这种额外耦合。

## 16.3 不冻结 ViT

B-B 的目标是直接修正共享 CLS 的检索几何。

因此：

- ViT 保持可训练；
- BNNeck 与分类器继续接受 CE；
- B-B 梯度直接回传共享 ViT；
- 不使用冻结教师；
- 不只训练辅助特征分支。

## 16.4 WHU-MARS-1000 与 1000-GD

**WHU-MARS-1000-GD 是 WHU-MARS-1000 模型的特定评测设置，不应默认作为独立训练数据集重新训练。**

当前数据类中 `FILTER_GD` 主要用于 Query/Gallery 筛选，训练集保持原完整数据。

因此：

- WHU-MARS-1000：正常完整训练并评测；
- WHU-MARS-1000-GD：使用相同 1000 训练得到的模型进行 GD 协议评测。

不能把训练数据也过滤成 GD 再与原论文表中的 UAD-GD 直接比较。

## 16.5 WHU-MARS-2337

该设置拥有不同的训练身份数量和数据分布。

需要在对应数据目录上从相同预训练方式独立训练。

不能直接将 1000 模型的训练身份分类器和 GPD Bank 当作 2337 的完整训练状态。

可共享已经确定的 B-B 机制、预算和温度配置，但训练身份数、分类器维度、原型库大小由数据集自动决定。

---

# 十七、原始 WRT 与 B-A 的对照实现

## 17.1 原 WRT 的数值风险

当前原代码的关键行为为：

```text
先计算 dist * mask 的最大值
再计算 exp(dist - max_v)
分母加 1e-6
```

对负样本使用负距离时，无效位置的零可能被当成最大 Logit。

如果忽略固定 epsilon，该偏移在归一化分子、分母中理论上可以抵消。

但它会削弱正常的数值稳定化效果，且当指数和很小时，固定 epsilon 可能使有效负样本权重之和小于 1。

因此不能简单声称原 WRT 所有情况下都计算错误。

## 17.2 B-A 的修改范围

B-A 仅修正负样本的有效 Softmax 数值归一化。

基本原则：

1. 无效负候选不参与最大值搜索；
2. 只在有效负样本上归一化；
3. 有效负样本权重和应接近 1；
4. 空集合有明确处理；
5. 不改 WRT 的距离与损失结构。

保持：

- 未归一化 CLS；
- 原欧氏距离；
- 原正样本规则；
- 原 SoftMarginLoss；
- 原损失权重。

不要在 B-A 中同时改变相机过滤或切换余弦相似度。

## 17.3 原 WRT 必须仍可使用

不要覆盖原实现后失去原 UAD 复现能力。

建议在代码中保留：

```text
原始 WRT
稳定修正 WRT
B-B Ranking
```

三个独立可选分支。

这样才能明确比较：

- 原始真实基线；
- 实现数值修正；
- 新排序机制。

---

# 十八、日志、诊断与一次性正确性检查

首次训练前需要一轮集中检查，但不需要反复增加新验证项目。

## 18.1 数据一致性

检查：

- 图像特征、PID、camera 长度一致；
- 模态拼接顺序正确；
- 同身份同相机样本被排除；
- 不同身份同相机样本被排除；
- Query 自身被排除；
- 有效 Query 比例合理。

## 18.2 权重数值

对每个有效 Query：

\[
\sum_pa_{qp}\approx1
\]

\[
\sum_jc_{qj}\approx1
\]

\[
\sum_{n\in N_{q,j}}d_{qn|j}\approx1
\]

\[
\sum_nb_{qn}\approx1
\]

同时确认：

- 权重非负；
- 无 NaN/Inf；
- 不存在无效候选获得正预算；
- \(\alpha_q\) 位于 0–0.5；
- 空集合不会导致非法 Softmax。

## 18.3 梯度正确性

检查：

- 排序损失对正相似度的直接偏导非正；
- 对负相似度的直接偏导非负；
- 动态预算没有梯度路径；
- `global_feat` 保留 B-B 梯度；
- B-B 可以向共享 ViT 产生非零梯度；
- BNNeck/分类器没有来自 B-B 的直接梯度；
- 原 CE、ProCA、GPD 梯度仍正常。

这些可通过少量独立梯度诊断完成，不要求在正式训练中持续执行昂贵的逐损失 backward。

## 18.4 必要的训练统计

建议每个 Epoch 或固定日志间隔输出：

```text
BB loss
valid_query_ratio
positive_count_mean
negative_identity_count_mean
top_gap_mean
top_error_ratio
alpha_mean
alpha_near_max_ratio
negative_identity_budget_stats
gradient_norm_BB（按较低频率）
```

不需要每一步保存完整的 \(N\times N\) 相似度矩阵。

## 18.5 有效 Query 覆盖不足时如何处理？

若只是少数 Query 没有跨相机正样本，按规则跳过即可。

但若大量 Query 都只有一个有效正样本，B-B1 的自适应多正样本预算实际上很难发挥作用。

这时问题已经涉及方法监督的有效覆盖，不能单纯提高 \(\lambda_B\) 作为补救。

应当根据实际统计决定是否需要调整候选定义或采样覆盖。

不过，首次方案**不预先改 PKM**。

---

# 十九、首次正式实验安排

## 19.1 主对照矩阵

| 实验 | 局部度量机制 | 研究目的 |
|---|---|---|
| 原始 UAD | 官方 WRT | 保留真实性能基准 |
| B-A | 稳定修正 WRT | 判断数值归一化问题的实际影响 |
| B-B0 | 统一 Pairwise、负身份预算、均匀正预算 | 检验新局部排序机制的整体价值 |
| B-B1 | B-B0 + Query 顶部风险预算 | 检验本研究核心自适应机制 |

B-B0 和 B-B1 必须严格一致的项目：

- 模型结构；
- 预训练初始化；
- 训练数据；
- PKM/DDP；
- 有效候选掩码；
- 负身份预算；
- margin；
- 所有温度；
- \(\lambda_B\)；
- CE、ProCA、GPD；
- 优化器和学习率；
- 训练时长；
- 评测协议。

两者唯一的方法变量为是否引入自适应正样本预算。

## 19.2 B-B0 的意义

B-B0 不仅仅是普通 Batch-All Pairwise Softplus。

它仍然包含：

- 归一化 CLS；
- 跨相机有效候选；
- 负身份风险；
- 负身份内部困难分配；
- 带前置温度的 Pairwise Loss。

因此，它检验的是新局部度量方案整体的有效性。

不能把 B-B0 相对于原 WRT 的所有收益都归因于单个改动。

## 19.3 B-B1 的意义

B-B1 相对于 B-B0 只改变正样本监督预算：

\[
\frac1{|P_q|}
\quad\longrightarrow\quad
\frac{1-\alpha_q}{|P_q|}
+\alpha_q\mathbf1[p=p_q^*]
\]

因此两者差异更有助于判断：

Query 顶部风险感知是否真的改善了 Rank-1 与其他指标的协调。

## 19.4 结果验收

首先报告完整 Mixed Gallery：

- mAP；
- Rank-1；
- Rank-5；
- Rank-10；
- mINP。

其次报告：

- 九种有向模态组合；
- Aerial→Ground、Ground→Aerial；
- 其他现有平台细分评测。

最后使用逐 Query 分析解释：

1. 原 UAD 成功、B-B 失败的 Query；
2. 原 UAD 失败、B-B 成功的 Query；
3. 最佳正样本相似度变化；
4. 最大负样本相似度变化；
5. 正样本整体排名变化；
6. 最后一个正确匹配排名变化。

尤其关注是否真正减少危险负身份侵入顶部，而不仅仅提高平均相似度间隔。

---

# 二十、完整前向与反向流程总图

```text
                      TRAINING

RGB images     NIR images       TIR images
     │              │               │
     └──────────────┼───────────────┘
                    │
         Batch 维拼接，非特征融合
                    │
                    ▼
              共享 ViT-B/16
                    │
                    ▼
             BN 前 CLS [N,768]
                    │
      ┌─────────────┼───────────────┬───────────────────┐
      │             │               │                   │
      ▼             ▼               ▼                   ▼
   BNNeck       B-B 新损失         ProCA                GPD
      │             │               │                   │
   Classifier    FP32 / L2      模态中心聚合         normalize CLS
      │             │               │                   │
      ▼             ▼               ▼                   ▼
     CE         Cosine [N,N]    身份中心 detach      PrototypeBank
                    │                                   │
               PID/Camera Mask                           │
                    │                                   ▼
               有效 Query / P / N                      GPD CE
                    │
          ┌─────────┴───────────┐
          │                     │
          ▼                     ▼
    顶部正负间隔 Δ        负身份 PID 分组
          │                     │
          ▼                     ▼
      Query 风险          负身份最大相似度
          │                     │
          ▼                     ▼
      正样本预算 a        身份预算 c
                                │
                                ▼
                         身份内图像预算 d
                                │
                                ▼
                         负样本预算 b
          │                     │
          └─────────┬───────────┘
                    │
          预算权重 a、b 停止梯度
                    │
          原始相似度 S 保留梯度
                    │
                    ▼
      所有有效正负 Pairwise Softplus
                    │
                    ▼
             Query 内加权求和
                    │
                    ▼
            DDP 有效 Query 归一化
                    │
                    ▼
                  λ_B L_B
                    │
      ┌─────────────┼───────────────┬───────────────────┐
      │             │               │                   │
      ▼             ▼               ▼                   ▼
     CE           B-B         0.01 ProCA              GPD
      │             │               │                   │
      └─────────────┼───────────────┴───────────────────┘
                    │
                    ▼
             总损失汇总
                    │
                    ▼
              AMP backward
                    │
                    ▼
           unscale + gradient clipping
                    │
                    ▼
           SGD 更新共享 ViT 等参数
                    │
                    ▼
            原 GPD PrototypeBank EMA 更新
                    │
                    ▼
                 下一 iteration


                       INFERENCE

                   单张 RGB/NIR/TIR 图像
                            │
                            ▼
                      同一个 ViT
                            │
                            ▼
                     BN 前 CLS 768D
                            │
                            ▼
                        L2Norm
                            │
                            ▼
                原官方 Gallery 排序/评测
```

---

# 二十一、最终实施原则与代码审查结论

## 21.1 必须修改的部分

1. 新增完整 B-B Ranking Loss；
2. 给局部损失传递展开后的 camera；
3. 在同一损失内部计算 Query 顶部风险、正负预算和 Pairwise Loss；
4. 正确停止预算梯度；
5. 在 FP32 中计算归一化相似度和排序目标；
6. 保证 \(\lambda_B\) 只施加一次；
7. 完成有效 Query 的 DDP 归一化；
8. 保留原 UAD、B-A、B-B0、B-B1 的独立实验配置。

## 21.2 不应修改的部分

1. 不增加任何 ViT 分支或特征融合；
2. 不改变最终 768D 检索 CLS；
3. 不将 CE 改为新的分类头；
4. 不修改 ProCA 的中心聚合和 stop-gradient；
5. 不修改 GPD 的 Bank 和温度；
6. 不引入教师模型；
7. 不增加新的困难身份分类损失；
8. 不改变 PKM 的基本采样策略；
9. 不使用跨卡真实图像候选；
10. 不采用 60+60 两阶段训练；
11. 不修改原始 Gallery 过滤及评测协议。

## 21.3 首次训练前的完成标准

只有以下事项是正式训练前的必要条件：

- 前向数据和标签顺序完全一致；
- 候选掩码与有效 Query 处理正确；
- 全部正负预算归一化正确；
- 新损失在 FP32 下数值稳定；
- 梯度能够正确回传到 BN 前 CLS 和共享 ViT；
- 原 CE、ProCA、GPD 仍正常训练；
- DDP 有效 Query 归一化正确；
- 完成一次 \(\lambda_B\) 校准；
- 正式训练配置与已有成功复现的四卡配置保持可比。

这些完成后，直接开始正式训练，不继续引入新的理论保护机制。

## 21.4 最终方案定位

B-B V1.0 的方法核心是：

**在 UAD 的统一 CLS 检索空间中，根据 Query 的真实顶部排序风险，协调最佳正样本与其他正确匹配的学习预算；同时根据当前负身份的最大图像混淆程度及其内部困难图像分布，分配受控负样本预算；最终通过统一的真实图像级 Pairwise Ranking Loss 更新共享 ViT。**

相对于原 WRT，它不再仅依赖加权正、负距离统计量，而是明确监督具体正负图像之间的排序错误。

相对于 WHV2，它不使用冻结教师、稀疏硬 Focus 或固定辅助融合。

相对于完整 SupRank，它保留较轻量的成对排序机制，并以当前 WHU 的真实 PKM 和 UAD 监督体系作为实施基础。

**最终创新不在于公式复杂度，而在于让正负样本监督预算更直接地对应当前检索错误，并且保持与 UAD 原有全局身份监督相互补充。**

本方案到此可以作为 B-B V1.0 的代码修改与首次正式训练实施依据。
