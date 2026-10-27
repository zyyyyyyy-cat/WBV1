# B-B V1.0：查询风险感知与负身份分层预算的统一排序学习方案

**研究任务：** WHU-MARS 多模态、多平台行人重识别  
**基础模型：** UAD（统一单分支 ViT-B/16）  
**设计状态：** V1.0 正式实验方案  
**核心改动：** 使用统一的 Query 级加权 Pairwise Ranking Loss 替换原 WRT  
**推理改动：** 无  
**新增可训练模块：** 无

---

# 一、研究目标与方案定位

## 1.1 总体研究目标

本研究以 WHU-MARS 数据集及原 UAD 模型为基础，研究如何改进多模态、多平台行人重识别的身份检索排序质量。

重点数据集与评测设置包括：

1. WHU-MARS-1000；
2. WHU-MARS-1000-GD；
3. WHU-MARS-2337。

主要评价指标包括：

- mAP；
- Rank-1；
- Rank-5；
- Rank-10；
- mINP。

**总体目标是在保持原有 UAD 有效能力的基础上，争取上述指标共同提升，并以相对各数据集原始基线提高至少 1.0 个百分点作为主要研究目标。**

同时关注九种 RGB、NIR、TIR 有向模态检索组合以及地面、航拍跨平台检索表现，避免通过牺牲某些关键方向换取有限的总体收益。

## 1.2 原 UAD 的局限与 B-B 的研究切入点

原 UAD 同时采用：

\[
L_{\mathrm{UAD}}
=
L_{\mathrm{CE}}
+
L_{\mathrm{WRT}}
+
0.01L_{\mathrm{ProCA}}
+
L_{\mathrm{GPD}}
\]

四类监督具有不同职责：

| 损失 | 主要职责 |
|---|---|
| CE | 对训练身份进行类别判别 |
| WRT | 在真实图像之间进行局部困难正负样本度量学习 |
| ProCA | 对齐同身份的不同模态中心 |
| GPD | 利用训练身份原型增强全局身份判别 |

这些机制已经具有合理的分工。

因此，B-B 不重新构建 CE、ProCA 或 GPD，也不增加第二套身份原型系统。

当前 B-B 的改进对象是 **WRT 所承担的真实图像级局部度量学习**。

原 WRT 先将正负样本距离分别聚合为加权统计量，再优化其距离间隔。

这种设计能够关注困难关系，但与最终完整 Gallery 中的身份排序并不完全等价。

B-B 希望进一步利用：

1. 哪些错误图像已经排在正确图像之前；
2. 最佳正样本是否正受到危险负样本威胁；
3. 其他正确匹配是否仍然被大量错误图像压制；
4. 哪些负身份及其高分图像需要获得更多排序监督。

最终使有限的训练梯度更有效地服务于未见身份的检索。

## 1.3 本方案的基本原则

B-B V1.0 确定以下设计原则。

**原则一：统一目标。**

不分别增加 Top Loss、Coverage Loss 和困难身份损失，而是将对应能力统一到一项 Query 级排序损失中。

**原则二：正样本预算随顶部风险变化。**

顶部排序危险时，适度加强最佳正样本；顶部较安全时，更多关注其他有效正样本。

**原则三：负身份具有受控、风险感知的预算。**

既关注真正危险的负身份，又避免某个身份的大量重复图像无约束地占据监督。

**原则四：动态权重负责分配预算，排序损失负责产生梯度。**

权重随当前特征动态更新，但在当前 step 中停止权重自身的梯度。

**原则五：保留 UAD 有效机制。**

不修改共享 ViT 结构、CE、ProCA、GPD、PKM 基本采样流程及最终检索特征。

**原则六：以实际检索收益为准。**

数学梯度的正确性是必要条件，但不等于性能保证。最终以完整 Gallery 的实际指标判断有效性。

---

# 二、基础模型与保留结构

## 2.1 原 UAD 网络

使用原始单分支 ViT-B/16 作为骨干网络。

训练输入包含三种模态：

- RGB；
- NIR，仓库中对应 `IR`；
- TIR，仓库中对应 `Thermal`。

图像分辨率保持原配置：

\[
256\times128
\]

每张图像经过共享 ViT，得到一个 768 维 CLS 特征：

\[
f_i=f_\theta(x_i)\in\mathbb R^{768}
\]

随后经过 BNNeck：

\[
f_i^{BN}=\mathrm{BNNeck}(f_i)
\]

用于身份分类：

\[
\mathrm{logits}_i=W_{\mathrm{cls}}f_i^{BN}
\]

这里需要严格区分：

- BN 前 CLS：`global_feat`；
- BN 后分类特征：`feat`；
- 分类输出：`cls_score`。

B-B 使用 **BN 前的 `global_feat`**。

不使用 BN 后特征，也不使用新增投影后的特征。

## 2.2 新增特征提取与融合方式

**B-B 不增加任何特征分支或融合模块。**

不使用：

- Parent/Child 分支；
- Patch Token 辅助特征；
- 模态专家；
- 固定比例 Global/Local 拼接；
- 额外的投影层；
- 推理期融合网络；
- 教师特征。

因此，本方案不存在需要学习的新特征融合系数。

所有训练监督仍然作用于原有共享 ViT 学到的身份表征。

这与此前 WV、WH、WHV2 的主要结构区别在于：

**B-B 改变的是 CLS 特征的学习方式，而不是改变最终检索特征的组成方式。**

## 2.3 最终推理特征

保持原 UAD 配置：

```yaml
TEST.NECK_FEAT: before
TEST.FEAT_NORM: yes
TEST.RE_RANKING: False
```

推理时首先提取 BN 前 CLS：

\[
f=f_\theta(x)
\]

然后执行 L2 归一化：

\[
\boxed{
z=\frac{f}{\|f\|_2+\epsilon}
}
\]

最终采用 \(z\in\mathbb R^{768}\) 计算检索距离。

不需要额外的模态标签、平台标签或者 B-B 权重计算。

**B-B 的全部新增计算仅存在于训练阶段。**

---

# 三、WHU 实际训练数据组织

## 3.1 原 PKM 采样

保持 WHU 原有 PKM 采样器。

根据 WHU-MARS-1000 的实际四卡训练日志：

| 参数 | 数值 |
|---|---:|
| GPU 数量 | 4 |
| 全局采样 batch | 256 个多模态采样单元 |
| 每 GPU 采样单元 | 64 |
| 每 GPU 身份数 | 16 |
| 每身份每模态图像数 | 4 |
| 模态数 | 3 |
| 每 GPU 实际图像数 | 192 |
| 每 GPU 不同负身份数 | 15 |
| 每 Query 最多其他同身份图像数 | 11 |

注意：配置中的 `IMS_PER_BATCH=256` 不代表全部 GPU 总共只有 256 张实际输入图像。

因为每个采样单元对应三个模态，四卡实际输入图像数为：

\[
256\times3=768
\]

每 GPU 为：

\[
64\times3=192
\]

B-B V1.0 使用每张 GPU 的本地 192 张图像构造排序关系。

暂时不跨 GPU Gather 图像特征，也不增加历史特征队列。

## 3.2 模态与标签排列

WHU 原模型使用：

```python
x = torch.cat(list(x), dim=0)
```

因此当前 GPU 的特征排列为：

\[
F=
[
F_{\mathrm{RGB}};
F_{\mathrm{NIR}};
F_{\mathrm{TIR}}
]
\]

假设每模态有 \(B=64\) 张图像，则：

\[
F\in\mathbb R^{192\times768}
\]

身份标签必须按同一顺序构造：

\[
Y=\operatorname{repeat}(target,3)
\]

相机标签：

\[
C=\operatorname{cat}
(c_{\mathrm{RGB}},c_{\mathrm{NIR}},c_{\mathrm{TIR}})
\]

其中：

\[
Y,C\in\mathbb Z^{192}
\]

**不得把特征解释成每身份 RGB、NIR、TIR 交错排列。**

不得使用未重复的 64 个身份标签去匹配 192 个图像特征。

原 CE、ProCA 和 GPD 的输入组织保持不变。

---

# 四、B-B 输入特征与归一化流程

## 4.1 特征来源

从原模型获取：

\[
F_{\mathrm{raw}}
=
\mathrm{global\_feat}
\in\mathbb R^{N\times768}
\]

其中本地：

\[
N=192
\]

## 4.2 数据精度

B-B 的全部相似度和权重计算使用 FP32：

\[
F_{32}=\operatorname{float32}(F_{\mathrm{raw}})
\]

这一步可以在局部关闭 AMP autocast 的上下文中完成。

不需要把 ViT 前向整体改成 FP32。

## 4.3 L2 归一化

对每张图像执行：

\[
\boxed{
z_i=
\frac{f_i}
{\max(\|f_i\|_2,\epsilon)}
}
\]

推荐采用 PyTorch `F.normalize`，并设置：

\[
\epsilon=10^{-12}
\]

得到：

\[
Z\in\mathbb R^{N\times768}
\]

归一化发生在**构造相似度矩阵之前**。

不得先计算原始欧氏距离，再对最终距离矩阵进行归一化替代这一操作。

## 4.4 相似度矩阵

计算：

\[
\boxed{
S=ZZ^\top
}
\]

其中：

\[
S\in\mathbb R^{N\times N}
\]

\[
s_{ij}=z_i^\top z_j
\]

理论上：

\[
s_{ij}\in[-1,1]
\]

该相似度与原 UAD 测试阶段归一化欧氏距离具有相同的排序关系：

\[
\|z_i-z_j\|_2^2=2-2s_{ij}
\]

因此 B-B 与最终检索采用一致的基础特征几何。

**相似度矩阵必须保留梯度。**

后续只有预算权重计算使用停止梯度的相似度副本。

---

# 五、构造有效 Query、正样本与负样本集合

这是正式计算损失前的第一项操作。

对任意 Query \(q\)，定义当前 GPU 内的候选集合。

## 5.1 有效正样本

\[
\boxed{
P_q=
\{p:y_p=y_q,\ p\ne q,\ c_p\ne c_q\}
}
\]

其中：

- \(y\)：训练身份 PID；
- \(c\)：实际相机 ID；
- \(p\ne q\)：排除自身；
- \(c_p\ne c_q\)：排除同相机候选。

该规则与 WHU 当前测试评测器的同相机排除口径保持一致。

注意：

跨相机不等于跨模态。

只要 PID 相同、Camera ID 不同，RGB、NIR、TIR 中的图像都可以构成有效正样本。

## 5.2 有效负样本

\[
\boxed{
N_q=
\{n:y_n\ne y_q,\ c_n\ne c_q\}
}
\]

同样排除与 Query 相同 camera ID 的候选。

即使 PID 不同，只要相机相同，本版 B-B 也不将其纳入直接检索排序关系。

这是一项明确的训练监督选择，并不改变 CE、GPD 等原始损失的训练样本范围。

## 5.3 有效 Query

仅当：

\[
|P_q|>0
\]

且：

\[
|N_q|>0
\]

时，Query 参与 B-B。

定义：

\[
\mathcal Q_{\mathrm{valid}}
=
\{q:|P_q|>0,\ |N_q|>0\}
\]

对于无有效正样本或无有效负样本的 Query：

- 不计算该 Query 的顶部风险；
- 不构造对应预算；
- 不参与 B-B 平均；
- 不影响该图像继续接受 CE、ProCA 和 GPD 监督。

当：

\[
|P_q|=1
\]

时，正样本权重自动等于 1，顶部自适应预算不会产生额外的正样本分配效果。

无需对这种情况单独引入损失。

## 5.4 有效性检查

首次正式训练前，应一次性核对：

- 有效 Query 比例；
- 有效正样本数量分布；
- 有效负身份数量分布；
- RGB、NIR、TIR 的有效 Query 覆盖；
- 不同相机及平台关系的覆盖。

如果多数 Query 都没有有效跨相机正样本，那么 B-B 的多正样本预算机制无法正常发挥作用。

此时需要优先修正候选定义或数据组织，而不是增加损失系数。

本方案第一版不预先修改 PKM。

---

# 六、Query 顶部排序风险

## 6.1 最佳正样本

对有效 Query：

\[
\boxed{
p_q^*
=
\arg\max_{p\in P_q}s_{qp}
}
\]

定义：

\[
s_q^+
=
s(q,p_q^*)
\]

这里必须只在有效正样本中选择。

不得包含 Query 自身。

否则，自身相似度接近 1，容易造成顶部风险判断失真。

## 6.2 最危险负样本

\[
\boxed{
n_q^*
=
\arg\max_{n\in N_q}s_{qn}
}
\]

定义：

\[
s_q^-
=
s(q,n_q^*)
\]

## 6.3 顶部间隔

\[
\boxed{
\Delta_q=s_q^+-s_q^-
}
\]

含义：

| \(\Delta_q\) | 状态 |
|---|---|
| 小于 0 | 当前有效 batch 的顶部身份排序错误 |
| 等于 0 | 正负第一名处于竞争边界 |
| 大于 0 | 最佳正样本领先 |
| 明显大于 0 | 当前顶部关系具有较大优势 |

\(\Delta_q\) 是当前训练 batch 的局部顶部间隔，并不直接等于测试完整 Gallery 的 Rank-1 状态。

## 6.4 顶部风险函数

定义：

\[
\boxed{
r_q=
\sigma\left(
-\frac{\Delta_q}{\tau_r}
\right)
}
\]

其中：

\[
\sigma(x)=\frac1{1+e^{-x}}
\]

本版采用：

\[
\boxed{\tau_r=0.05}
\]

作为首次训练初值。

这里将顶部风险的中心设为：

\[
\Delta_q=0
\]

而不是：

\[
\Delta_q=m
\]

原因是：

- \(\Delta_q=0\) 对应实际第一名的正负交换边界；
- \(m\) 对应 Pairwise Ranking 所希望保留的安全间隔。

这两个概念不应强制绑定。

## 6.5 正样本额外顶部预算

定义：

\[
\boxed{
\alpha_q=
\alpha_{\max}r_q
}
\]

固定：

\[
\boxed{
\alpha_{\max}=0.5
}
\]

因此：

\[
0<\alpha_q<0.5
\]

它不是可训练参数，而是固定的预算上限。

\(\alpha_q\) 根据当前 Query 排序状态在每个 step 动态重新计算。

当顶部错误严重时，\(\alpha_q\) 接近 0.5。

当顶部较安全时，\(\alpha_q\) 接近 0。

在：

\[
\Delta_q=0
\]

时：

\[
\alpha_q=0.25
\]

**计算 \(\alpha_q\) 时停止梯度。**

---

# 七、正样本监督预算

## 7.1 权重定义

对每个有效正样本 \(p\)：

\[
\boxed{
a_{qp}
=
\frac{1-\alpha_q}{|P_q|}
+
\alpha_q\mathbf1[p=p_q^*]
}
\]

其中：

\[
\mathbf1[\cdot]
\]

为指示函数。

## 7.2 设计含义

第一项：

\[
\frac{1-\alpha_q}{|P_q|}
\]

保证所有有效正样本具有基础监督预算。

第二项：

\[
\alpha_q\mathbf1[p=p_q^*]
\]

根据当前顶部风险增加最佳正样本的关注度。

两者属于同一个预算分配过程，不是两个独立损失。

## 7.3 权重性质

满足：

\[
a_{qp}\ge0
\]

\[
\boxed{
\sum_{p\in P_q}a_{qp}=1
}
\]

最佳正样本的权重为：

\[
a_{qp^*}
=
\frac{1-\alpha_q}{|P_q|}+\alpha_q
\]

其他正样本为：

\[
a_{qp}
=
\frac{1-\alpha_q}{|P_q|}
\]

当：

\[
|P_q|=11,\quad\alpha_q=0.5
\]

时：

\[
a_{qp^*}\approx0.5455
\]

其余十个正样本分别约为：

\[
0.0455
\]

因此，即使顶部严重错误，其他正样本仍保留整体监督预算。

## 7.4 为什么不增加困难正样本 Softmax？

Pairwise Softplus 本身已经根据每个正样本被错误候选超过的程度产生不同梯度。

如果再增加一套基于最低正样本相似度的困难权重，可能重复放大某些低信息质量样本。

因此，B-B V1.0 不增加独立困难正样本权重。

正样本侧只使用：

**均匀覆盖 + 最佳正样本的风险感知额外预算。**

## 7.5 梯度规定

\[
a_{qp}
\]

在当前 step 中视为常数。

停止梯度仅作用于权重，不作用于正样本相似度：

\[
s_{qp}
\]

也不冻结最佳正样本对应的 ViT 特征。

---

# 八、负身份级困难度

## 8.1 按 PID 组织负候选

对 Query \(q\)，将有效负样本按训练身份划分。

设共有：

\[
J_q
\]

个不同负身份。

第 \(j\) 个负身份的有效图像集合为：

\[
N_{q,j}
\]

其数量：

\[
K_{q,j}=|N_{q,j}|
\]

并满足：

\[
N_q=\bigcup_{j=1}^{J_q}N_{q,j}
\]

不同 PID 的集合互不重叠。

这里使用的只是当前 batch 已有的 PID 标签。

不创建新的负身份原型，不使用 GPD Bank 计算身份困难度。

## 8.2 身份顶部威胁

定义：

\[
\boxed{
h_{qj}
=
\max_{n\in N_{q,j}}s_{qn}
}
\]

这个量表示第 \(j\) 个错误身份中，最容易与 Query 混淆的真实图像的相似度。

因为：

\[
\max_jh_{qj}
=
\max_{n\in N_q}s_{qn}
\]

因此，真正占据当前错误候选顶部的身份一定具有最大的 \(h_{qj}\)。

这里采用 Max 是为了识别危险身份，而不是让单张负样本独自承担整个损失。

**身份困难度仅用于预算计算，不直接构成额外的 Max Triplet Loss。**

## 8.3 身份风险分布

定义：

\[
v_{qj}
=
\frac{
\exp(h_{qj}/\tau_w)
}{
\sum_{k=1}^{J_q}
\exp(h_{qk}/\tau_w)
}
\]

实现中使用数值稳定的 Softmax。

首轮采用：

\[
\boxed{\tau_w=0.10}
\]

此温度用于负身份和身份内部的困难分配。

它与顶部风险温度及实际排序损失温度相互独立。

---

# 九、负身份预算

## 9.1 风险感知身份预算

定义：

\[
\boxed{
c_{qj}
=
\frac{1-\beta}{J_q}
+
\beta v_{qj}
}
\]

其中首轮固定：

\[
\boxed{\beta=0.5}
\]

第一部分负责身份覆盖。

第二部分根据顶部混淆风险提高危险负身份的预算。

## 9.2 预算归一化

满足：

\[
c_{qj}\ge0
\]

\[
\boxed{
\sum_{j=1}^{J_q}c_{qj}=1
}
\]

并且：

\[
\frac{1-\beta}{J_q}
\le c_{qj}
\le
\frac{1-\beta}{J_q}+\beta
\]

在：

\[
\beta=0.5
\]

时，每个负身份都保留基础预算，同时最危险身份可以获得额外预算。

## 9.3 为什么不用严格身份均等？

如果某个负身份已经有一张图像排在所有正样本之前，而其余身份明显安全，那么让所有负身份获得相同预算，可能削弱顶部纠错力度。

因此，本版使用受控的风险感知预算。

但也不允许危险身份无限吸收梯度。

---

# 十、负身份内部图像预算

## 10.1 困难图像分布

对于第 \(j\) 个身份内部的图像：

\[
\pi_{qn|j}
=
\frac{
\exp(s_{qn}/\tau_w)
}{
\sum_{x\in N_{q,j}}
\exp(s_{qx}/\tau_w)
}
\]

相似度较高的图像具有更高权重。

## 10.2 均匀与困难分布结合

定义：

\[
\boxed{
d_{qn|j}
=
\frac{1-\rho}{K_{q,j}}
+
\rho\pi_{qn|j}
}
\]

固定：

\[
\boxed{\rho=0.5}
\]

因此身份内部：

- 50% 的显式预算对应均匀覆盖；
- 50% 的显式预算根据困难图像分布分配。

## 10.3 归一化性质

\[
d_{qn|j}\ge0
\]

\[
\boxed{
\sum_{n\in N_{q,j}}d_{qn|j}=1
}
\]

## 10.4 最终负图像预算

将身份预算与身份内部预算相乘：

\[
\boxed{
b_{qn}
=
c_{qj}d_{qn|j}
}
\]

满足：

\[
\boxed{
\sum_{n\in N_{q,j}}b_{qn}=c_{qj}
}
\]

以及：

\[
\boxed{
\sum_{n\in N_q}b_{qn}=1
}
\]

这意味着一个负身份的所有有效图像，共同分享该身份获得的预算。

## 10.5 梯度规定

以下变量均停止梯度：

- \(h_{qj}\)；
- \(v_{qj}\)；
- \(c_{qj}\)；
- \(\pi_{qn|j}\)；
- \(d_{qn|j}\)；
- \(b_{qn}\)。

这里的停止梯度是针对它们作为预算计算结果的用途。

不能因此将原始相似度矩阵 \(S\) 或特征 \(Z\) 整体停止梯度。

---

# 十一、统一 Pairwise Ranking Loss

## 11.1 正负排序错误

对 Query \(q\)、有效正样本 \(p\)、有效负样本 \(n\)，定义：

\[
e_{qpn}=s_{qn}-s_{qp}
\]

当：

\[
e_{qpn}>0
\]

说明错误候选排在正确候选之前。

## 11.2 排序安全间隔

引入：

\[
m\ge0
\]

希望满足：

\[
s_{qp}\ge s_{qn}+m
\]

首轮推荐：

\[
\boxed{m=0.05}
\]

这是归一化余弦相似度空间中的固定间隔，不是原始欧氏距离间隔。

该值属于首次训练初值，不应表述为已经在 WHU 上验证的最优 margin。

## 11.3 单对排序损失

采用：

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

其中：

\[
\operatorname{softplus}(x)=\log(1+e^x)
\]

推荐：

\[
\boxed{\tau_\ell=0.05}
\]

为什么在 Softplus 前乘 \(\tau_\ell\)？

因为此时：

\[
\frac{\partial\ell_{qpn}}
{\partial(s_{qn}-s_{qp})}
=
\sigma\left(
\frac{m+s_{qn}-s_{qp}}{\tau_\ell}
\right)
\]

其导数位于：

\[
(0,1)
\]

温度主要控制困难关系附近的梯度平滑范围，而不会通过 \(1/\tau_\ell\) 自动放大最大导数。

## 11.4 Query 级统一损失

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

注意：

**这里的正负预算是在 Pairwise Loss 之外相乘。**

不要先把所有正样本的相似度聚合成一个加权均值，再与负样本均值计算 Softplus。

两种计算不是同一个目标。

本方案使用的是每一对真实正负关系的损失，再依据权重聚合。

## 11.5 Batch 级排序损失

定义：

\[
Q_v=|\mathcal Q_{\mathrm{valid}}|
\]

最终：

\[
\boxed{
L_B
=
\frac1{Q_v}
\sum_{q\in\mathcal Q_{\mathrm{valid}}}
L_B(q)
}
\]

这里采用 Query 平均，而不是将所有正负三元组直接统一求平均。

每个有效 Query 的正样本预算和为 1，负样本预算和也为 1。

因此，单个 Query 的损失不会仅因为候选数量更多，就线性增大。

---

# 十二、反向梯度的数学性质

这是本方案最核心的机制说明。

## 12.1 权重停止梯度后的正样本梯度

定义：

\[
u_{qpn}
=
\frac{m+s_{qn}-s_{qp}}{\tau_\ell}
\]

对某个正样本相似度：

\[
\boxed{
\frac{\partial L_B(q)}
{\partial s_{qp}}
=
-a_{qp}
\sum_{n\in N_q}
b_{qn}\sigma(u_{qpn})
}
\]

因此：

\[
\frac{\partial L_B(q)}
{\partial s_{qp}}\le0
\]

梯度下降对应提高正确匹配的相似度。

## 12.2 负样本梯度

对某个负样本相似度：

\[
\boxed{
\frac{\partial L_B(q)}
{\partial s_{qn}}
=
b_{qn}
\sum_{p\in P_q}
a_{qp}\sigma(u_{qpn})
}
\]

因此：

\[
\frac{\partial L_B(q)}
{\partial s_{qn}}\ge0
\]

梯度下降对应降低错误匹配的相似度。

## 12.3 直接相似度梯度上界

由于：

\[
\sum_pa_{qp}=1
\]

\[
\sum_nb_{qn}=1
\]

并且：

\[
0<\sigma(u)<1
\]

得到：

\[
\left|
\frac{\partial L_B(q)}
{\partial s_{qp}}
\right|
\le a_{qp}
\]

\[
0\le
\frac{\partial L_B(q)}
{\partial s_{qn}}
\le b_{qn}
\]

对第 \(j\) 个负身份：

\[
\boxed{
\sum_{n\in N_{q,j}}
\left|
\frac{\partial L_B(q)}
{\partial s_{qn}}
\right|
\le c_{qj}
}
\]

这里的上界是**单个 Query 的相似度变量层面**的性质。

不代表整个 ViT 参数梯度范数也具有相同上界。

## 12.4 动态权重为什么停止梯度？

如果允许权重反向传播：

\[
\nabla(a b\ell)
=
ab\nabla\ell
+
b\ell\nabla a
+
a\ell\nabla b
\]

后两项会使模型通过改变预算权重本身来优化损失。

这可能产生额外、难以直观解释的优化方向。

B-B V1.0 只保留：

\[
ab\nabla\ell
\]

因此权重负责在本 step 中选择监督重点，而排序损失负责产生实际梯度。

**下一 step 仍会使用更新后的模型特征重新计算所有动态权重。**

所以这不是静态困难样本权重。

## 12.5 为什么相似度层面的单调性不等于最终 Rank-1 保证？

因为：

\[
s_{ij}=z_i^\top z_j
\]

而：

\[
z_i=\operatorname{normalize}(f_\theta(x_i))
\]

同一图像会同时作为多个 Query、正样本和负样本参与损失。

不同 Query 的梯度最终在共享 ViT 参数上叠加。

一次参数更新可能使某些正样本相似度提高，也可能间接改变其他样本关系。

因此，B-B 只能保证其直接排序损失的偏导方向合理，不能从单步偏导推出测试阶段每个 Query 的 Rank-1 单调提高。

---

# 十三、完整前向数据流程

下面按照 WHU 实际训练流程描述一次完整 iteration。

## Step 1：PKM 采样与图像增强

从原 WHU 数据加载器取得：

- 三模态图像；
- 身份 PID；
- 相机 ID。

保持原有：

- Resize；
- Random Horizontal Flip；
- Padding；
- Random Crop；
- Normalize；
- Random Erasing。

不新增数据增强。

## Step 2：共享 ViT 前向

三模态图像按照原顺序拼接：

\[
X=[X_{\mathrm{RGB}};X_{\mathrm{NIR}};X_{\mathrm{TIR}}]
\]

经过共享 ViT：

\[
F_{\mathrm{raw}}=f_\theta(X)
\]

得到 BN 前 CLS。

## Step 3：原 CE 分支

执行：

\[
F_{\mathrm{BN}}=\mathrm{BNNeck}(F_{\mathrm{raw}})
\]

\[
\mathrm{logits}=W_{\mathrm{cls}}F_{\mathrm{BN}}
\]

计算：

\[
L_{\mathrm{CE}}
\]

保持原 CE 计算及权重不变。

## Step 4：B-B 特征归一化

在 FP32 中：

\[
Z=\operatorname{normalize}(F_{\mathrm{raw}})
\]

随后计算：

\[
S=ZZ^\top
\]

## Step 5：构造有效样本掩码

依据完整展开后的 PID 与 camera ID：

- 排除自身；
- 排除全部同相机候选；
- 分别提取正负集合；
- 判断 Query 是否有效。

## Step 6：计算 Query 顶部风险

在有效集合中选择：

\[
p_q^*,n_q^*
\]

得到：

\[
\Delta_q
\]

以及：

\[
\alpha_q
\]

## Step 7：计算正样本预算

根据：

\[
a_{qp}
=
\frac{1-\alpha_q}{|P_q|}
+\alpha_q\mathbf1[p=p_q^*]
\]

得到当前 Query 的正样本预算。

## Step 8：计算负身份预算

将有效负候选按 PID 分组。

对每个负身份：

\[
h_{qj}=\max_ns_{qn}
\]

然后计算：

\[
c_{qj}
\]

## Step 9：计算身份内图像预算

每个负身份内部根据相似度计算：

\[
d_{qn|j}
\]

最终：

\[
b_{qn}=c_{qj}d_{qn|j}
\]

## Step 10：停止预算权重梯度

对：

\[
a_{qp},b_{qn}
\]

停止梯度。

不得对用于排序损失的：

\[
s_{qp},s_{qn}
\]

停止梯度。

## Step 11：计算所有有效 Pairwise Loss

构造：

\[
m+s_{qn}-s_{qp}
\]

通过带温度前置系数的 Softplus 得到单对损失。

乘以：

\[
a_{qp}b_{qn}
\]

得到 Query 级排序损失。

## Step 12：Query 平均

对全部有效 Query 求平均，得到：

\[
L_B
\]

多 GPU 情况下采用后述的全局有效 Query 归一化方式。

## Step 13：计算原 ProCA

直接使用原始 BN 前：

\[
F_{\mathrm{raw}}
\]

维持原代码的模态中心对齐过程。

不使用经过 B-B 权重处理的特征。

## Step 14：计算原 GPD

继续使用：

- 原始 CLS；
- 原训练身份标签；
- 原 PrototypeBank；
- 原温度；
- 原原型更新机制。

不把 B-B 的困难身份预算注入 GPD。

## Step 15：汇总总损失

\[
\boxed{
L_{\mathrm{total}}
=
L_{\mathrm{CE}}
+
\lambda_B L_B
+
0.01L_{\mathrm{ProCA}}
+
L_{\mathrm{GPD}}
}
\]

其中 \(\lambda_B\) 为训练前固定的损失系数。

## Step 16：反向传播与参数更新

沿用原有 AMP、梯度裁剪、优化器和学习率调度。

## Step 17：更新原 GPD Bank

按照原 UAD 的流程，在优化器更新之后，以无梯度方式更新 PrototypeBank。

不由 B-B 直接修改原型。

---

# 十四、完整反向梯度流程

反向传播从总损失开始：

\[
L_{\mathrm{total}}
\]

分别形成四条梯度路径。

## 14.1 CE 梯度路径

\[
L_{\mathrm{CE}}
\rightarrow
W_{\mathrm{cls}}
\rightarrow
\mathrm{BNNeck}
\rightarrow
F_{\mathrm{raw}}
\rightarrow
\mathrm{ViT}
\]

更新：

- 身份分类器参数；
- BNNeck 可训练参数；
- 共享 ViT 参数。

## 14.2 B-B 梯度路径

\[
L_B
\rightarrow
\ell_{qpn}
\rightarrow
S
\rightarrow
Z
\rightarrow
F_{\mathrm{raw}}
\rightarrow
\mathrm{ViT}
\]

具体来说：

首先，Pairwise Ranking 对正负相似度产生梯度。

然后，相似度矩阵：

\[
S=ZZ^\top
\]

把梯度传递到 Query 与候选图像两侧的归一化特征。

再经过 L2 归一化的反向链式法则：

\[
Z=\operatorname{normalize}(F_{\mathrm{raw}})
\]

返回 BN 前 CLS。

最后更新共享 ViT。

**B-B 不直接更新 BNNeck 和身份分类器。**

但 ViT 更新后，其他监督分支下一次前向使用的特征也会改变。

## 14.3 ProCA 梯度路径

\[
L_{\mathrm{ProCA}}
\rightarrow
\text{不同模态身份中心}
\rightarrow
F_{\mathrm{raw}}
\rightarrow
\mathrm{ViT}
\]

保留原代码中的 detached 全局中心目标。

## 14.4 GPD 梯度路径

\[
L_{\mathrm{GPD}}
\rightarrow
\text{图像—身份原型相似度}
\rightarrow
F_{\mathrm{raw}}
\rightarrow
\mathrm{ViT}
\]

PrototypeBank 是非梯度更新的状态缓冲区，不是通过优化器训练的分类器。

原型在反向传播之后按原机制更新。

## 14.5 共享 ViT 的总梯度

最终：

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

这些梯度在共享 ViT 参数上相加。

梯度合并后，再执行原有全局梯度裁剪。

这里不增加：

- PCGrad；
- GradNorm；
- 动态多任务损失平衡；
- 第二套梯度投影；
- 梯度保护分支。

---

# 十五、参数与超参数的完整设定

## 15.1 B-B V1.0 首轮参数表

| 参数 | 含义 | 首轮设定 | 是否可训练 |
|---|---|---:|---|
| 特征维度 \(D\) | BN 前 CLS 维度 | 768 | 否，结构常数 |
| \(\epsilon\) | L2 归一化稳定项 | \(10^{-12}\) | 否 |
| \(m\) | Pairwise 余弦安全间隔 | **0.05** | 否 |
| \(\tau_r\) | Query 顶部风险平滑温度 | **0.05** | 否 |
| \(\tau_w\) | 负身份及身份内困难权重温度 | **0.10** | 否 |
| \(\tau_\ell\) | Pairwise Softplus 排序温度 | **0.05** | 否 |
| \(\alpha_{\max}\) | 最大额外顶部正样本预算 | **0.5** | 否 |
| \(\beta\) | 负身份风险预算比例 | **0.5** | 否 |
| \(\rho\) | 身份内部困难图像预算比例 | **0.5** | 否 |
| \(\lambda_B\) | B 相对其他监督的梯度强度系数 | 一次性梯度标定后固定 | 否 |
| \(a_{qp}\) | 当前 Query 的正样本预算 | 每 step 动态计算 | 停止梯度 |
| \(b_{qn}\) | 当前 Query 的负样本预算 | 每 step 动态计算 | 停止梯度 |
| \(\Delta_q\) | 当前有效 batch 的顶部相似度间隔 | 每 step 动态计算 | 仅用于预算，不反传 |
| \(S\) | 真实图像余弦相似度矩阵 | 每 step 重新计算 | **参与反传** |

以上 \(m\) 与温度为首次运行的工程初值，并非已由 WHU 实测确定的最优参数。

首轮不进行多维网格搜索。

## 15.2 原 UAD 参数

以下保持原实际复现设置。

对于已经完成的 WHU-MARS-1000 四卡基线：

| 参数 | 实际值 |
|---|---:|
| `MAX_EPOCHS` | 120 |
| `NUM_INSTANCE` | 4 |
| `IMS_PER_BATCH` | 256，实际四卡全局采样单元 |
| `BASE_LR` | 0.032 |
| 优化器 | SGD |
| Warm-up | 5 Epoch |
| CE 系数 | 1.0 |
| ProCA 系数 | 0.01 |
| GPD 系数 | 1.0 |
| 原 Metric Loss 系数 | 1.0 |
| GPD 温度 | 0.03 |
| GPD momentum 配置 | 0.2 |
| 随机种子 | 1234 |

注意：

仓库 `configs/UAD.yml` 中保存的单卡基础配置与四卡训练日志的最终生效配置不同。

复现四卡实验时，应以实际运行日志记录的配置为准，而不是只读取 YAML 文件中的默认值。

---

# 十六、\(\lambda_B\) 的确定方法

这是首次正式训练前需要完成的一次性数值标定。

## 16.1 为什么不能直接按损失值定系数？

原 WRT 使用未归一化特征欧氏距离。

B-B 使用归一化余弦相似度。

两者的损失值和对原始 CLS 的梯度映射都不同。

因此：

\[
L_{\mathrm{WRT}}\approx L_B
\]

并不意味着：

\[
\|\nabla_F L_{\mathrm{WRT}}\|
\approx
\|\nabla_F L_B\|
\]

也不意味着两者对共享 ViT 的影响相同。

## 16.2 标定规则

在与正式训练一致的初始化、预处理、PKM batch 和计算精度下，固定随机种子，抽取一组具有代表性的真实训练 batch。

建议先使用 **12 个 PKM batch** 完成一次性校准。

对每个 batch，在不更新模型参数的情况下分别计算：

\[
g_W^{(b)}
=
\left\|
\frac{\partial L_{\mathrm{WRT}}}
{\partial F_{\mathrm{raw}}}
\right\|_F
\]

\[
g_B^{(b)}
=
\left\|
\frac{\partial L_{B0}}
{\partial F_{\mathrm{raw}}}
\right\|_F
\]

并记录 GPD 的 CLS 梯度范数作参考。

其中 \(L_{B0}\) 使用与 B-B1 相同的负身份规则，但正样本采用均匀预算。

对各 batch 求中位数：

\[
G_W=\operatorname{median}_b(g_W^{(b)})
\]

\[
G_B=\operatorname{median}_b(g_B^{(b)})
\]

以有效实现的 WRT 梯度规模作为参考，确定：

\[
\boxed{
\lambda_B=\frac{G_W}{G_B+\epsilon_g}
}
\]

这里 \(\epsilon_g\) 只是防止除零的数值稳定项，不能用于掩盖接近零的异常梯度。

标定过程中同时记录原 WRT 与修正数值实现后的 WRT 梯度，优先以数值正确、权重实际归一化的 WRT 作为尺度参考。

若计算得到极端异常的比例，应首先检查是否发生有效 Query 缺失、梯度为零或归一化错误，而不是直接将异常系数投入训练。

## 16.3 标定结果的使用

标定完成后：

- 将实际计算得到的数值写入正式实验配置；
- 整个 120 Epoch 训练过程中保持不变；
- B-B0 与 B-B1 使用同一个 \(\lambda_B\)；
- 不在训练过程中动态调整；
- 不根据测试指标重新搜索该系数。

这样既控制训练强度，也避免把动态梯度平衡引入研究目标。

**当前无法预先写出经过真实数据验证的 \(\lambda_B\) 数值；它必须由上述确定性校准过程得到。**

这不是未完成的方法设计，而是一个已定义清楚的数值标定步骤。

---

# 十七、FP32、AMP 与计算顺序

## 17.1 保留混合精度训练

原模型的 ViT 前向与其他原始损失可继续使用现有 AMP 流程。

B-B 的以下运算在 FP32 中完成：

1. CLS 转换；
2. L2 归一化；
3. 余弦相似度矩阵；
4. 顶部风险；
5. 负身份困难度；
6. 负身份与身份内部 Softmax；
7. Pairwise Softplus；
8. Query 损失聚合。

## 17.2 不要提前停止整个特征梯度

错误做法：

```python
z = F.normalize(global_feat.detach().float(), dim=1)
```

因为这会阻止 B-B 更新共享 ViT。

正确原则是：

```python
z = F.normalize(global_feat.float(), dim=1)
```

随后只将用于计算预算的相似度副本停止梯度。

## 17.3 Softmax 数值稳定性

所有有效候选 Softmax 都应使用数值稳定实现。

不要在掩码位置填零后，直接把该位置参与最大值搜索。

对无效位置，应在 Softmax 前屏蔽。

空集合必须先排除，不允许对空集合直接执行 Max 或 Softmax。

## 17.4 Pairwise Softplus

采用稳定的 `softplus` 实现。

不手动使用：

```python
log(1 + exp(x))
```

替代稳定算子，以免较大输入引起指数溢出。

---

# 十八、四卡 DDP 的损失归一化

B-B 首轮使用本地训练图像关系，不进行跨卡特征 Gather。

但是，四张 GPU 的有效 Query 数量可能不同。

如果每卡分别执行：

\[
\frac{S_r}{N_r}
\]

再由 DDP 平均参数梯度，就不一定等价于全部 GPU 有效 Query 的整体平均。

定义第 \(r\) 张 GPU：

\[
S_r=\sum_{q\in Q_r^{valid}}L_B(q)
\]

\[
N_r=|Q_r^{valid}|
\]

GPU 数量：

\[
W=4
\]

通过分布式求和得到：

\[
N_{\mathrm{all}}=\sum_rN_r
\]

每张 GPU 使用：

\[
\boxed{
L_{B,r}
=
\frac{W S_r}
{N_{\mathrm{all}}}
}
\]

DDP 平均各卡梯度后，等价于对全部有效 Query 的损失求平均。

注意：

- 只对有效 Query 数量执行跨卡求和；
- 不需要 Gather 全部图像特征；
- Query 数量没有梯度；
- 保持 B-B 的本地候选定义不变。

如果全局没有任何有效 Query，则将 B-B 设为与当前计算图连接的零损失，并保持所有 Rank 行为一致，避免除零或分布式流程不一致。

原 CE、ProCA 和 GPD 的 DDP 计算保持原有方式。

---

# 十九、原 ProCA、GPD 与 B-B 的协同

## 19.1 ProCA 保持原实现

ProCA 使用 BN 前 CLS，对每个身份和模态内的特征归一化并求中心，再计算不同模态中心的对齐损失。

其全局中心目标保留原有停止梯度行为。

B-B 不改变：

- ProCA 的模态分组；
- 每模态每身份的实例数量；
- ProCA 的中心归一化；
- ProCA 的损失系数；
- ProCA 的梯度路径。

## 19.2 GPD 保持原实现

GPD 使用原始 CLS 的归一化特征与全局身份原型计算相似度。

继续使用：

\[
\tau_{\mathrm{GPD}}=0.03
\]

保留：

- 原型维度 768；
- 原型数量等于当前训练身份数量；
- 原始 momentum 更新规则；
- 原有跨卡原型统计同步；
- 原始初始化与更新顺序。

B-B 的负身份困难度不调用 GPD 原型。

GPD 的目标是全局身份级判别，B-B 则直接处理真实图像间的排序错误。

## 19.3 是否存在梯度竞争？

存在可能性。

但 B-B V1.0 不额外增加梯度冲突补偿模块。

其原因是，当前已经通过：

- 统一损失；
- 明确预算归一化；
- 停止权重梯度；
- 单对梯度幅度控制；
- 固定 \(\lambda_B\) 校准；

对新增监督作出了必要约束。

后续如果正式结果证明存在严重的优化冲突，再根据证据讨论。

---

# 二十、训练策略

## 20.1 不采用 HiHR 式 60+60 阶段切换

B-B 不增加新的网络分支，因此没有必须分阶段学习的结构性理由。

原 HiHR 在阶段切换时涉及优化器、学习率和监督变化，多项因素共同影响了后续训练表现。

B-B V1.0 不重复这一安排。

## 20.2 使用连续 120 Epoch 联合训练

首次正式实验：

- 保持 ImageNet ViT 初始化；
- 保持原 UAD 优化器；
- 保持原 UAD 学习率日程；
- 保持 120 Epoch；
- 保持原 PKM；
- 保持原 CE、ProCA、GPD；
- 从训练开始使用 B-B 替代 WRT；
- 不额外启用第二阶段 Warm-up；
- 不冻结共享 ViT。

## 20.3 不采用 WHV2 的冻结教师设计

WHV2 证明，冻结 Global 并通过固定辅助融合进行纠错，不一定能形成足够的检索增量。

B-B 直接更新共享 CLS 表征。

其目标是改变原特征空间中不够合理的身份排序，而不只是添加辅助特征修正结果。

## 20.4 不改变推理结构

全部新增预算、困难身份分组和 Pairwise Loss 在训练结束后不再使用。

测试仍运行：

\[
x\rightarrow\mathrm{ViT}\rightarrow\mathrm{CLS}
\rightarrow\mathrm{L2Norm}
\rightarrow\mathrm{Retrieval}
\]

---

# 二十一、关键争议点的最终审查

## 21.1 为什么选择 Pairwise，而不是直接采用 SupRank？

Pairwise Ranking 已经能够对真实正负排序错误产生明确监督。

当前 B-B 进一步通过 Query 状态和负身份组织改善监督预算。

完整 SupRank 可能更贴近 AP 与 Recall@k，但同时引入排名近似、候选集合和计算开销问题。

第一版先研究轻量排序目标是否有效，B-C 保留为后续具有明确证据时的拓展方向。

## 21.2 为什么顶部风险使用 \(\Delta=0\)？

因为它直接对应 Rank-1 正负第一名的交换边界。

安全 margin 由 Pairwise Loss 单独承担。

这样避免让顶部预算机制与排序间隔参数相互绑定。

## 21.3 为什么负身份用 Max，不用 LogMeanExp？

Max 更直接地反映某个错误身份是否存在高度危险的真实图像。

由于 Max 仅用于停止梯度的预算计算，并不意味着最终只对最难图像训练。

身份内部的 Softmax 和均匀预算仍保留多图像监督。

## 21.4 为什么身份间和身份内可以共用 \(\tau_w\)？

这两层都用于负样本预算，而且当前采用 Max 身份困难度，不再存在原先 LogMeanExp 使用相同温度导致的普遍精确退化关系。

为了减少首轮自由度，允许共用 \(\tau_w\)。

但顶部风险与实际排序损失使用独立温度。

## 21.5 为什么不增加单独的 Top Loss？

因为最佳正样本和危险负身份已经通过统一目标中的预算获得额外关注。

再增加独立 Top Loss 会重复强化类似关系，增加优化复杂度。

## 21.6 为什么不对全部困难正样本额外分配权重？

Pairwise Loss 已经根据正样本与不同负样本之间的相对排序错误产生不同梯度。

再增加独立困难正样本 Softmax 可能过度强化视觉信息不足的极端正样本。

## 21.7 为什么不只保留跨模态候选？

因为最终 Mixed Gallery 既包含跨模态，也包含同模态候选。

WAV1 的失败诊断显示，危险错误匹配并不只来自跨模态关系。

所以 B-B 使用统一跨相机候选，不额外限定必须跨模态。

## 21.8 为什么不增加新的负身份 Bank？

GPD 已经提供训练身份级全局判别。

B-B 需要的是当前真实图像的困难排序关系。

再次增加身份 Bank 可能重复监督，也增加状态维护与更新滞后问题。

## 21.9 为什么不立即扩大跨卡候选？

当前首轮要验证 B-B 的关系组织和预算是否有效。

跨卡特征会同时改变候选身份数量、梯度传播和计算成本。

如果首次实验显示损失机制合理但有效困难负身份覆盖不足，再独立研究跨卡候选更有价值。

## 21.10 为什么不直接保证所有指标一定提升？

因为排序损失是在有限训练身份和候选集合上学习的代理目标。

即使其梯度方向合理，也不能严格保证未见身份的完整 Gallery Rank-1、AP 和 INP 同时改善。

这是需要真实实验检验的核心研究假设，而不是公式自身能够证明的结论。

---

# 二十二、首次训练前的必要检查

本方案不需要再开展无休止的设计验证。

正式训练前只执行一次基础正确性与数值校准。

检查范围限定为：

| 检查项 | 必须满足的要求 |
|---|---|
| 特征来源 | BN 前 CLS，768 维 |
| 特征排列 | RGB、IR、Thermal 分段排列 |
| 标签排列 | 与模态拼接顺序完全一致 |
| 自身样本 | 完全排除 |
| 同相机候选 | 按当前评测规则排除 |
| 有效 Query | 正负集合非空 |
| 正样本预算 | 每个有效 Query 的权重和为 1 |
| 负身份预算 | 身份预算和为 1 |
| 身份内部预算 | 每个身份内部图像权重和为 1 |
| 总负样本预算 | 每个有效 Query 的负权重和为 1 |
| 停止梯度 | 只停止预算权重的梯度 |
| 排序梯度 | 正负相似度仍参与梯度 |
| 数值精度 | B-B 核心计算 FP32 |
| 损失数值 | 无 NaN、Inf |
| 梯度 | 符号与范数合理 |
| \(\lambda_B\) | 一次性标定并固定 |
| DDP | 全局有效 Query 归一化正确 |
| 原监督 | CE、ProCA、GPD 未被意外修改 |
| 推理 | 完全保持原 UAD 768D CLS |

所有检查通过后，直接进入首次正式训练。

不为每一个数值细节单独安排新一轮完整训练。

---

# 二十三、正式实验与消融设计

## 23.1 四个主要实验

| 实验 | 改动 | 作用 |
|---|---|---|
| UAD | 原始官方 WRT | 保留真实基线 |
| B-A | 仅修正原 WRT 的负样本权重数值归一化 | 判断原实现问题的影响 |
| B-B0 | 新 Pairwise Loss + 负身份预算 + 均匀正样本预算 | 判断新局部排序机制的整体价值 |
| **B-B1** | B-B0 + Query 顶部风险感知正样本预算 | 判断自适应顶部预算是否具有额外收益 |

B-B0 中：

\[
a_{qp}=\frac1{|P_q|}
\]

其余所有内容与 B-B1 一致。

两者必须保持：

- 相同数据集；
- 相同初始化；
- 相同训练预算；
- 相同优化器与学习率；
- 相同候选掩码；
- 相同负身份预算；
- 相同温度和 margin；
- 相同 \(\lambda_B\)；
- 相同评测协议。

## 23.2 B-A 的实现范围

B-A 仅修改 WRT 的有效负样本权重归一化。

保留原 WRT 的：

- 特征输入；
- 欧氏距离；
- 正样本规则；
- SoftMarginLoss；
- 损失系数；
- 训练流程。

不把 B-A 的数值修复与新的 Pairwise 目标混为一谈。

特别注意：原 WRT 无效位置取到零，主要问题在于数值稳定化与固定 epsilon 可能影响实际权重；不能直接断言原实现所有情况下都算错。

## 23.3 主要验收指标

以完整官方 Gallery 为主要判断依据：

1. Rank-1；
2. mAP；
3. Rank-5；
4. Rank-10；
5. mINP。

此外保留：

- 九种模态组合；
- 跨平台方向；
- Rank-1 新成功与新失败 Query；
- 最佳正样本与最危险负样本间隔；
- 多正样本排名变化。

训练诊断仅用于解释结果，不重新定义最终性能目标。

---

# 二十四、最小必要代码修改范围

结合当前 [WHU 仓库](https://github.com/zyyyyyyy-cat/WHU)，建议实现集中于以下位置。

| 文件或模块 | 修改目的 |
|---|---|
| `loss/` | 新增 B-B 统一排序损失实现 |
| `loss/make_loss.py` | 增加 B-B 类型选择，保证原 WRT 不被重复调用 |
| `processor/processor.py` | 接入 B-B，传入 PID 和 camera ID，汇总总损失 |
| `config/defaults.py` | 增加 B-B 相关超参数及默认值 |
| `configs/` | 提供 B-A、B-B0、B-B1 独立实验配置 |
| `loss/wrt_loss.py` | 仅在 B-A 实验中提供受控数值修正 |
| 训练日志 | 增加必要的 B-B 权重、风险和有效 Query 统计 |

不修改：

- ViT 主干结构；
- 模型推理 `forward`；
- BNNeck；
- 原身份分类器；
- ProCA 数学定义；
- GPD 数学定义；
- PrototypeBank；
- 原评测协议；
- PKM 的基本采样结构。

若现有训练接口已经提供完整展开的 camera ID，应直接复用，不新增不必要的数据传递。

---

# 二十五、最终完整方案总结

B-B V1.0 的完整逻辑为：

**第一阶段：身份特征提取。**

RGB、NIR、TIR 图像进入同一个 ViT-B/16，得到原始 768 维 CLS。

**第二阶段：统一检索空间。**

对 BN 前 CLS 进行 L2 归一化，在 FP32 中构造真实图像的余弦相似度矩阵。

**第三阶段：有效排序候选。**

根据 PID 和 camera ID 构造跨相机有效正负集合，排除自身与无效 Query。

**第四阶段：Query 风险感知。**

通过当前最佳正样本与最危险负样本的相似度差，判断顶部排序风险，并决定最佳正样本获得多少额外预算。

**第五阶段：负身份与图像预算。**

按 PID 组织错误候选，以身份内部最大相似度判断真实顶部威胁，再在身份之间、身份内部图像之间分配受控监督预算。

**第六阶段：统一排序监督。**

将所有有效正负关系的 Pairwise Softplus 按正负预算加权，在 Query 内聚合，再对有效 Query 求平均。

**第七阶段：反向传播。**

预算权重停止梯度；排序相似度保留梯度，通过 L2 归一化映射回原始 CLS，并直接更新共享 ViT。

**第八阶段：与 UAD 联合优化。**

CE、ProCA、GPD 保留原有功能与参数，B-B 替换 WRT，按固定校准系数参与原有总损失。

**第九阶段：连续训练。**

沿用原 UAD 四卡、120 Epoch、SGD、PKM 和学习率流程，不引入第二阶段、冻结教师或新的推理结构。

**第十阶段：统一检索。**

训练完成后，仍然只提取 BN 前 768 维 CLS，执行 L2 归一化并使用原 WHU 评测协议。

---

## 最终研究定位

**B-B 的核心创新不在于提出又一种 Softplus，而在于将 Query 顶部排序风险、多正样本监督覆盖和危险负身份竞争，统一为一个可解释、受控的真实图像级梯度分配机制。**

其希望解决的是：

原 UAD 已经具有较强的身份判别和困难度量能力，但局部距离监督未必将足够的学习压力分配给最终检索中最关键的错误排名关系。

通过 B-B，希望在不增加推理复杂度的前提下，改善共享 CLS 的身份检索几何，使未见身份的 Rank-1、mAP、Rank-5/10 和 mINP 尽可能共同提升。

**B-B V1.0 到此可以作为首次正式训练的完整候选方案。** 除一次性候选有效性检查、数值检查和 \(\lambda_B\) 梯度校准外，不需要再增加新的网络模块或损失机制。

正式训练结果将决定该方法是否真正解决了 WHU-MARS 中的检索排序问题，而不是由公式的完整性提前决定结论。

---

**主要代码依据**

- [WHU 复现代码](https://github.com/zyyyyyyy-cat/WHU)
- [原 WRT 实现](https://github.com/zyyyyyyy-cat/WHU/blob/main/loss/wrt_loss.py)
- [训练处理流程](https://github.com/zyyyyyyy-cat/WHU/blob/main/processor/processor.py)
- [模型结构](https://github.com/zyyyyyyy-cat/WHU/blob/main/model/make_model.py)
- [PKM/DDP 采样](https://github.com/zyyyyyyy-cat/WHU/blob/main/datasets/sampler_ddp.py)
- [检索评测协议](https://github.com/zyyyyyyy-cat/WHU/blob/main/utils/metrics.py)

**相关研究参考**

- UAD / WHU-MARS，CVPR 2026；
- *Optimization of Rank Losses for Image Retrieval*，IEEE TPAMI 2025，SupRank；
- *MINER: Modeling Inter- and Intra-Person Variance for Reidentification*，2026 年公开研究。

上述论文为研究问题和机制选择提供参考；B-B 的具体预算公式及其有效性仍属于本研究提出、尚待正式训练验证的方案。
