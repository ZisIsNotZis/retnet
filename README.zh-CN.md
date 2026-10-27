<p align="center">
  <img src="assets/logo.svg" alt="retnet logo" width="112" height="112">
</p>

<h1 align="center">retnet</h1>

<p align="center"><strong>从零实现的字节级 RetNet 实验</strong></p>

<p align="center">
  一个无 softmax 的 retention 层，从零写起；小到可以读完，完整到可以训练。
</p>

<p align="center"><a href="README.md">English</a> | <strong>简体中文</strong></p>

---

## 为什么有这个项目

这是一个人的研究档案，不是产品。它只想回答一个很窄的问题：在极小的
字节级语言模型里，能不能用一个便宜、可并行、推理时又能写成循环的 mixer
——对 `K ⊙ V` 的累加和施加可学习的指数衰减——替代 attention 或空洞卷积？
如果可以，retention 和因果卷积又该怎么叠？

实验在 notebook 里用 Ray Tune 跑过，然后就停在了那里。没有论文，没有和
baseline 的对比，也没有保存下来的结果。留下的是完整的工作轨迹：层库、
模型定义、搜索配置，以及产生它们的 notebook。

## 快速开始

这里没有包、CLI、测试套件，也没有 requirements 文件。最快的入口是参考
notebook。

```bash
git clone https://github.com/ZisIsNotZis/retnet.git
cd retnet
pip install torch numpy "ray[tune]" jupyterlab
jupyter lab ray3.ipynb
```

`ray3.ipynb` 是最完整的一条路径：retention block → 字节级模型 → 自带
`train()` 的训练循环（Adam + OneCycleLR，bf16 autocast）→ 三轮 Ray Tune
ASHA 搜索。在 CUDA 机器上从头到尾运行即可。

**数据不在仓库里。** 每个模型都通过 `nn.Dat` 读取原始字节语料；它会
memory-map 文件，并写出 `path.map` 词表 sidecar：

```python
from nn import Dat

dat = Dat("py.txt", 32, 256)  # (文件, epoch 数, batch, 上下文长度)
```

把占位路径换成你自己的字节文件即可。一个简单的起点是转成小写的源代码
语料；`code.sh` 就是作者用来生成这类语料的脚本。生成的 `.map`、
`ray_results/` 和 `log.txt` 都已加入 git ignore。

### 核心思路

retention block 是用普通层拼出来的。去掉外围模型后，大致就是这样：

```python
from nn import *

model = Seq(
    Embedding(68, 64),
    Ret(64),          # 带学习衰减的 retention
    Linear(64, 68),   # 下一个字节的 logits
)
```

## 现在能做什么

- **从零搭出 retention mixer**：`nn.Decay` 实现线性与对数空间的指数衰减
  累加；`nn.Ret` 把它接成带因果投影的多头 key/value block。
- **不依赖框架地组合模型**：`Seq`、`Add`、`Mul`、`Dot`、`Res`、`Stack`
  让 notebook 读起来接近架构图。
- **训练字节级语言模型**：memory-map 字节数据、bf16 autocast、Adam、
  OneCycleLR，以及 loss/accuracy 记录。
- **低成本搜索结构**：用 Ray Tune ASHA 搜索宽度、头数、卷积核大小和
  decay 范围，每个 trial 约占 0.33–0.5 张 GPU。
- **探索 retention + 卷积的混合结构**：纯 retention 堆叠、因果卷积分支，
  以及带 stride/dilation 的 U-Net 式下采样/上采样。
- **读到完整的研究轨迹**：28 个 notebook、0 个保存的输出，以及一个
  420 行的层库——它就是 [`cznn`](https://github.com/ZisIsNotZis/cznn) 的前身。

**这里没有的东西：** `py.txt` 语料、checkpoint、loss 曲线、最佳配置、
测试、CI、requirements 文件。这些 notebook 记录的是“试过什么”，不是
可复现的 benchmark。

## 架构

```text
字节语料
   │
   ├─ Dat                 memory-map + 字节词表 sidecar
   │
   └─ Embedding(68, d)
        │
        └─ N × [ retention（Decay + 因果投影）+ 因果卷积 ]
             │
             └─ Linear(d, vocab) → 下一个字节的 logits
```

| 路径 | 是什么 |
| --- | --- |
| `nn.py` | 420 行的组合式层库：`Ret`、`Decay`、`Dat`、`Seq`/`Add`/`Mul`/`Dot`/`Res`、参数量估算工具和因果卷积工具。 |
| `nnn.py` | 同一批想法的早期分支，142 行；没有 notebook 引用它，作为研究轨迹的一部分保留。 |
| `ray3.ipynb` | **从这里开始。** 最完整的 notebook：`Ret`、`Model`、自带的 `train()`，以及三轮 ASHA 搜索。 |
| `ray1.ipynb`、`ray2.ipynb` | 同一路线上的 `Blk`/`Blks` 模块族（retention 与因果卷积分支并行）、`get_blks`，以及连续五轮 ASHA 搜索。 |
| `ray3cnn-Copy5.ipynb`、`Untitled9.ipynb` | 最大的 retention + 卷积混合实验场，包含 `Cov`/`Covd`/`Covu` 和 `torch.compile(torch.jit.trace(...))` 训练循环。 |
| `Untitled-Copy1.ipynb` | 相邻的一条线：字节级 CNN 语言模型、随机结构搜索和文本生成采样器。 |
| `code.sh` | 个人脚本：把源代码文件收集成小写字节语料。 |
| `docs/project-status.md` | 证据、边界，以及这个档案目前的分类。 |

retention 的想法来自 *Retentive Network: A Successor to Transformer for Large
Language Modeling*（Sun et al., 2023），这里是按作者自己的层风格重新推导的，
不是从论文里照搬的代码。

## 如何开发

- **notebook 就是工作单元。** 没有 build，没有测试命令，也没有要安装的包。
  打开 notebook，用你自己的语料运行即可。
- **层库留在本地。** notebook 依赖 `from nn import *`；`nn.py` 留在仓库里，
  这些 import 才能继续解析。正式发布的规范版本是
  [`cznn`](https://github.com/ZisIsNotZis/cznn)。
- **部分 notebook 缺名字是正常的。** 一些 notebook 依赖更早的 notebook 或
  根本不存在的辅助函数（例如 `ray3.ipynb` 调用了未定义的 `size_select`）。
  它们按原样保留，没有被整理成教程。
- **不要提交生成物。** `*.map`、`ray_results/`、`log.txt`、`__pycache__/`
  和 notebook checkpoint 都已被忽略。语料和权重不该进仓库。
- **依赖写在快速开始里。** 那份清单只覆盖 `ray3` 这条路径。其他 notebook
  还会用到 `matplotlib`、`pandas`、`tqdm`、`hyperopt`、`transformers`、
  `numba`、`llvmlite`、`greenlet`、`lark`、`smac`、`scipy`；打开哪个就装
  哪个。这里没有 requirements 文件，也没有 lockfile 替你固定版本。

## 如何贡献

这是一个休眠的研究档案，但修正和可复现性改进欢迎提交。请开 issue 说明
对应的 notebook 和 cell，或者提交范围集中的 pull request，尽量保持档案
忠于当时的运行状态。简短说明见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 路线图

### 已实现

- 线性与对数空间的 retention（`Decay`），支持因果位移。
- `nn.py` 中的 `Ret` block 和组合式模型搭建。
- 字节级 `Dat` 加载器，以及词表 sidecar。
- 自带 bf16 autocast、Adam、OneCycleLR 的训练循环。
- 针对模型宽度、头数、卷积核和 decay 范围的 Ray Tune ASHA 搜索。
- retention + 因果卷积、U-Net 式混合结构的实验。
- 档案文档和 AGPL-3.0-only 许可证。

### 计划中（尚未排期）

目前没有在做的功能。如果这个档案被重新启动，最值得做的下一步是：

- 用一份小的公开字节语料做一个最小可运行示例。
- 为核心 `ray3` 路径补一份固定版本的环境说明。
- 写一篇 retention、因果卷积和混合结构的对比，附上可复现搜索得到的真实曲线。
- 保存最佳配置和 loss 曲线，让结论可以被检查。

## 未来愿景

让它成为一个关于 retention 字节级语言模型的小而可信的参考：一条命令就能
在公开语料上训练一个可读的模型，配置可复现，真实曲线和完整研究轨迹放在
一起。层库已经在那里了；缺的是写下来的结论和一条可复现的路径。

## 与 `cznn` 的关系

这个目录里的 `nn.py` **就是**后来发布为
[`cznn`](https://github.com/ZisIsNotZis/cznn) 的库（`cznn/__init__.py`）。
正式版本以 `cznn` 为准；这里的文件保留下来，只是为了让 notebook 里没改过的
import 继续工作。`diff` 只会看到 `cznn` 添加的来源说明头。

## 许可证

[AGPL-3.0-only](LICENSE)。retention 的想法来自上面引用的 RetNet 论文；
实现和实验记录归作者所有。
