# AutoBM25 —— 零标注、零调参的 BM25 超参数自适应

[![PyPI](https://img.shields.io/pypi/v/autobm25.svg)](https://pypi.org/project/autobm25/)
[![Python](https://img.shields.io/pypi/pyversions/autobm25.svg)](https://pypi.org/project/autobm25/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/SaturnAura/AutoBM25/blob/main/LICENSE)
[![Rust MSRV](https://img.shields.io/badge/rust-1.71%2B-orange.svg)](https://github.com/SaturnAura/AutoBM25/blob/main/rust/Cargo.toml)
[![CI](https://github.com/SaturnAura/AutoBM25/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SaturnAura/AutoBM25/actions/workflows/ci.yml)

> [English README](https://github.com/SaturnAura/AutoBM25/blob/main/README.md)

![AutoBM25 logo](https://raw.githubusercontent.com/SaturnAura/AutoBM25/main/logo/logo.png)

> 输入任意语料的**统计特征** → **O(1) 大字典查表**直接得到最优 BM25 超参数（k1 / b / k3 / δ / IDF）；查不到就启发式兜底。16 个 BEIR 数据集验证：**16/16 不劣于默认参数、NDCG@10 中位数 +18.0%（均值 +30.5%）、留一语料集词典命中率 92.0%**。

> 😄 或许我们应该叫 **BM26**？——参数都能自适应了，版本号也自适一下（开个玩笑，BM25 还是 BM25）。

---

## 为什么需要它

BM25 是信息检索最常用的排序模型，但它的**最优超参数随语料变化极大**——短文本、长文本、代码、法律文书、问答社区，各自的最优 k1/b 完全不同。实际工程却永远用 `k1=1.2, b=0.75` 的默认值，因为调参需要带标注的验证集，而新语料往往**没有标注**。

AutoBM25 换了一条路：**最优参数是语料统计结构的函数**，而统计结构不需要标注就能测量。测量出 21 个归一化词典特征，查表即得参数——零标注、零调参、常数时间。

## 亮点

- 🗂️ **21 个词典特征**（15 个语料特征 + 6 个查询侧特征）：覆盖文档长度分布、词频冗余度、词汇增长曲线与查询统计，每个特征都有明确的机制假设
- ⚡ **O(1) 参数大字典**：`特征向量 → 最优参数` 最近邻查表，含**分布外（OOD）防护**与**启发式回退**；48 个条目随包提交（16 个基准语料 + 32 个跨域拼接语料），每条都在来源语料的全部查询上验证过不劣于默认参数
- 🔧 **k3 查询词频饱和项**：消融试验确认有效，长查询场景相对提升最高 **+19.9%**
- 🧩 **多语料拼接（Mixing）**：把已有语料组合拼接生成"跨域样本"，主动在特征空间补点，词典命中率 **70.6% → 92.0%**
- 🚀 **规模验证到 884 万文档**（msmarco），全自研 numpy 向量化 BM25/BM25+ 引擎，超大数据集流式子采样、相关标注全保留
- 🦀 **Rust/PyO3 加速**：分词建索引阶段提供编译扩展，本次基准中**单遍端到端快 5.2–12.4×**，分数一致（可选，纯 Python 依旧可用）

## 数值提升（16 个 BEIR 数据集）

**原版 BM25（默认参数）vs AutoBM25（自适应）—— MRR@10 / NDCG@10 / Recall@100**

每个单元格是 `原版 BM25 → AutoBM25`，两边用**同一份语料、同一批查询、同一份 qrels** 计算
（默认参数：k1=1.2, b=0.75, k3=0, δ=0, rsj IDF）。16 个数据集全部有提升，没有任何一个低于它所在行的基线。

| 数据集           | MRR@10                     | NDCG@10                    | Recall@100                 |
| ---------------- | -------------------------: | -------------------------: | -------------------------: |
| arguana          | 0.2036 → 0.2411            | 0.3158 → 0.3654            | 0.9090 → 0.9573            |
| climate-fever    | 0.2480 → 0.3049            | 0.1778 → 0.2241            | 0.4551 → 0.5306            |
| dbpedia-entity   | 0.6674 → 0.6919            | 0.3863 → 0.4158            | 0.4779 → 0.5152            |
| fever            | 0.5983 → 0.8467            | 0.6105 → 0.8322            | 0.9007 → 0.9536            |
| fiqa             | 0.1727 → 0.2050            | 0.1387 → 0.1624            | 0.3429 → 0.3799            |
| hotpotqa         | 0.8131 → 0.8376            | 0.6478 → 0.6803            | 0.7700 → 0.7830            |
| msmarco          | 0.4024 → 0.4162            | 0.4400 → 0.4548            | 0.7490 → 0.7530            |
| nfcorpus         | 0.4600 → 0.4740            | 0.2633 → 0.2738            | 0.1983 → 0.2027            |
| nq               | 0.1594 → 0.1957            | 0.1774 → 0.2266            | 0.5937 → 0.6587            |
| scidocs          | 0.2228 → 0.2596            | 0.1215 → 0.1444            | 0.2936 → 0.3260            |
| scifact          | 0.5033 → 0.5574            | 0.5388 → 0.5945            | 0.8116 → 0.8334            |
| trec-covid       | 0.5936 → 0.7832            | 0.3322 → 0.5011            | 0.0573 → 0.0764            |
| trec-covid-beir  | 0.6107 → 0.7982            | 0.3394 → 0.5113            | 0.0575 → 0.0761            |
| trec-covid-v2    | 0.4079 → 0.7941            | 0.2127 → 0.5041            | 0.0495 → 0.0933            |
| vihealthqa       | 0.3824 → 0.3921            | 0.3676 → 0.3758            | 0.6247 → 0.6289            |
| webis-touche2020 | 0.4931 → 0.7170            | 0.2226 → 0.3901            | 0.4336 → 0.5250            |
| **平均**         | **0.4337 → 0.5322**        | **0.3308 → 0.4160**        | **0.4828 → 0.5183**        |

16 个语料的 NDCG@10 相对提升：**中位数 +18.0%，均值 +30.5%**（最低 +2.2%，最高 +137.0%）。

这些数字是怎么来的（读表前值得知道）：

- 每个字典条目都是在该语料的**全部查询**上从候选池里选出来的，而候选池里始终包含 BM25 默认参数
  —— 因此一个条目不可能是它来源语料上"比默认更差"的参数；表格量的就是上线路径本身（查字典，
  未命中则回退启发式）。
- dbpedia-entity、climate-fever、fever、hotpotqa、msmarco 五个语料被降采样到 20–30 万篇文档。
  采样时会**保留全部标注相关文档**，所以绝对数值不能与全量 BEIR 榜单直接比较；每一格的两个数字
  都取自同一份采样、同一批查询与标注。
- 如果语料不在字典里，参数来自最近的条目或启发式。把目标语料自己的条目排除后，字典仍覆盖
  **92.0%** 的目标（16 个语料 + 34 个跨域混合语料，共 50 个）。

## 快速开始

```bash
pip install autobm25
```

源码目录安装使用 `pip install .`。如果检测到 Cargo，安装过程会尝试编译可选
Rust 扩展；否则自动安装功能完整的纯 Python 后端。

然后直接把语料交给它 —— 传一个字符串列表就够了，不需要学任何文件格式：

```python
from autobm25 import AutoBM25

corpus = [
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Automatic parameter tuning improves retrieval effectiveness.",
]
retriever = AutoBM25()
retriever.index(corpus)                  # 直接传字符串列表即可，不依赖任何文件
print(retriever.params)                  # 该语料自动选出的超参数
retriever.search("parameter tuning", top_k=5)
```

`corpus` 可以是任何"列表状"的东西：字符串、`{"id", "text"}` 记录、dataframe 的某一列、从接口
拉回来的文档。整个过程不写盘，也不要求任何目录结构。`index(corpus, queries=[...])` 再把查询
一起传进去，查询侧统计量（查询长度、查询词在语料中的覆盖、idf）会并入同一个特征向量。

语料已经在磁盘上？`autobm25 --dataset dataset/XXX` 会对目录跑同一套流程，加 `--interactive`
就变成输入查询的交互模式——CLI 只是这套 API 的便利封装，不是硬性要求。`autobm25 --demo`
连文件都不用准备。

CLI 会打印本数据集自适应选择的 BM25 超参数（如 `k1=0.5 b=0.495 k3=0.7 δ=0.97 idf=smoothed`）。

参数字典随包内置（`autobm25/data/param_dictionary.json`），安装后开箱即用，无需先跑任何实验。

想先看效果？`autobm25 --demo` 会索引一份内置的小语料（十篇自己编的文档，分别讲冲煮、骑行和检索），打印自动选出的超参数和几条查询的排序结果，不需要任何数据集目录；`autobm25 --demo --interactive` 可以直接对这份语料输入你自己的查询，`python examples/quickstart.py` 则是同一件事的可复制脚本（语料是一份编出来的客服 FAQ）。

## Rust 加速（PyO3）

检索内核同时提供了一份 Rust 实现，通过 **PyO3** 编译成 Python 扩展模块：热点循环跑在编译代码里，而对外接口、特征定义和排序结果与上文完全一致。

该扩展是**可选**的。发布的多平台 wheel 默认包含它；从源码安装时，有 Cargo
就尝试编译，没有就自动回退到纯 Python。`autobm25.BACKEND` 会告诉你当前用的是
哪一个，设置 `AUTOBM25_NO_RUST=1` 可以强制走 Python。

### 哪些阶段走了 Rust

整条 pipeline 是：加载 → 分词建索引 → 特征 → 预测 → 检索。扩展覆盖中间的三个阶段以及打分；
剩下两个用 Python 是刻意的选择。

| 阶段                               | 实现           | 成本                                                                                           |
| ---------------------------------- | -------------- | ---------------------------------------------------------------------------------------------- |
| 加载文档与查询                     | Python         | I/O + `json` 解析；几 GB 的语料文件要几秒到几十秒——这正是加载器**没有**进 Rust 的原因          |
| 分词 + 倒排索引（`build_index`）   | **Rust**       | 一遍扫过语料，同时把特征统计也收齐                                                             |
| 特征向量（`extract_features`）     | **Rust**       | 直接读取这一遍的统计量，不再重复分词                                                           |
| 参数预测（查字典 + 启发式）        | Python         | O(1)：48 条 × 21 维，**每个语料 0.1–0.5 ms**，config/字典都带缓存                              |
| `search` / `search_many`           | **Rust**       | posting list 累加、释放 GIL、每个工作线程独立 scorer                                           |
| 评测指标（CLI 批量模式）           | Python         | 对返回的排序做 numpy 平均，毫秒级                                                              |

加载器留在 Python，是因为它本质是 I/O + JSON 解析——在几 GB 的语料上它就是墙钟时间的大头，
换成 Rust 也不会让磁盘读得更快。参数预测留在 Python 则是另一个原因：它本身就是常数时间查表，
向量化没有收益。

### Rust 扩展替换掉了什么

瓶颈**不在打分**——那部分本来就已经是 numpy 向量化。瓶颈是**一个遍历全语料每个 token 的
Python 循环，每个 token 做一次字典操作，而且整段跑了两遍**：一遍算统计特征，一遍建倒排索引。

| 阶段           | Python                                                           | Rust                                                             |
| -------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| 分词与词频统计 | `str.lower().split()`，每个 token 一次字典操作                   | 单遍扫描语料；词项放在扁平表里，只有词项首次出现时才哈希字符串   |
| 统计特征       | 重新分词每个文档，并且在内存里为**所有**文档各保存一份字典       | 直接复用建索引那一遍的副产品——语料只分词**一次**                 |
| 倒排索引       | 对 Python 侧攒出来的 posting 列表逐词项 `np.asarray`             | postings 在同一遍里直接建好                                      |
| BM25 打分      | 已向量化，但用 `np.add.at` 散射累加，且每个查询都要对全数组排序  | 沿 posting 列表累加、每个查询零分配、释放 GIL、每线程一个 scorer |

因此 `index()` 对整个语料只分词一次而不是两次，这是收益最大的一处改动。纯 Python 版的 `BM25Engine` 原样保留，仍然可用。

### 用户需要装 Rust 吗？不需要

PyO3 是**编译期**工具：Rust 代码只在打包时编译一次，编译产物就是一个普通的 Python 扩展模块，最终分发出去的是它。使用者只需照常导入，不需要编译器、不需要工具链、不需要任何额外步骤：

```python
import autobm25                       # 有编译好的扩展就自动加载

retriever = autobm25.AutoBM25()       # 底层就是 Rust
retriever.index(corpus)               # 整语料只分词一遍
retriever.search("parameter tuning", top_k=5)

print(autobm25.BACKEND)               # "rust"
```

给别人用的时候，一个 wheel 就够了：`pip install <wheel>`（或从索引 `pip install autobm25`，
pip 会自动挑匹配系统/CPU 的那个），因为扩展已经在包里。`abi3-py39` 让构建矩阵保持可控——
**同一平台上**一份二进制就服务 CPython 3.9 及之后所有版本，所以 wheel 按操作系统/架构
（`win_amd64`、`manylinux_x86_64`、`macosx_arm64` …）分，而不是按 Python 版本分。做不到
"一个 wheel 通吃所有平台"：编译扩展天然与平台绑定，这也是源码仓库里没有编译产物时打出来的
是 `py3-none-any` 纯 Python wheel 的原因。

`import autobm25` 绝不会因为缺二进制或二进制不匹配而失败：`autobm25/_accel.py` 有扩展就用、
没有就回退纯 Python，只是慢约 7 倍。`AUTOBM25_NO_RUST=1` 运行时强制走 Python，
`AUTOBM25_SKIP_RUST_BUILD=1` 让安装阶段跳过编译，`autobm25.backend_info()` 会告诉你当前
用的是哪一个。

从源码安装则取决于有没有 `cargo`：有工具链时 `pip install .` 会自动编译，也可以手动
`python rust/build_pyext.py`（加 `--check` 会顺带验证能否导入）；没有工具链也照样能用，
只是退回纯 Python。发布构建可设置 `AUTOBM25_REQUIRE_RUST=1`，缺少扩展时立即失败，
避免静默产出不含加速器的 wheel。工具链会依次尝试默认的、再是已安装的
`*-windows-gnu`，也可以用 `--toolchain <name>` 直接指定。

### 实测加速

2026-11-09 在 Windows 上实测：12 个逻辑核，CPython 3.11.5，numpy 1.26.4，
scipy 1.12.0。同一台机器、同一份数据、同一个进程，`top_k=100`；Python 一列是
同一个包加上 `AUTOBM25_NO_RUST=1`。

| 数据集        | 文档数  | token 数 | 查询数 | 建索引+特征                          | 检索（全部查询）                     | 端到端                               |
| ------------- | ------: | -------: | -----: | -----------------------------------: | -----------------------------------: | -----------------------------------: |
| arguana       | 8,674   | 145 万   | 1,406  | 2.27 s → 0.24 s（**9.6×**）          | 3.87 s → 0.22 s（**17.6×**）         | 6.38 s → 0.52 s（**12.4×**）         |
| scifact       | 5,183   | 111 万   | 1,109  | 1.16 s → 0.13 s（**8.8×**）          | 0.39 s → 0.06 s（**6.2×**）          | 1.59 s → 0.23 s（**6.8×**）          |
| nfcorpus      | 3,633   | 85 万    | 3,237  | 0.97 s → 0.11 s（**9.3×**）          | 0.35 s → 0.07 s（**4.9×**）          | 1.50 s → 0.29 s（**5.2×**）          |
| trec-covid-v2 | 129,179 | 2717 万  | 50     | 41.27 s → 4.22 s（**9.8×**）         | 0.37 s → 0.05 s（**7.1×**）          | 42.88 s → 5.26 s（**8.1×**）         |

*端到端* = 加载 + 参数预测 + 建索引 + 特征 + 检索。若按拆分调用（`extract_features` 与 `build_index` 各自分词一遍）计，则是 **3.7–8.9×**。

怎么读这张表：随着语料增大，Python 逐 token 的字典操作成本通常会让差距更明显。检索提升在长查询场景最明显（arguana，17.6×），因为旧的 `np.add.at` 散射累加在那里最慢；批量路径会释放 GIL 并吃满所有核。

### 一致性

在同一份语料上跑两套后端，逐项比对特征向量、选出的超参数、每个查询的完整排序，以及
MRR@10 / NDCG@10 / Recall@100，结论是一致的。在 scifact、nfcorpus、arguana、trec-covid-v2 上：

- 特征值差异约 `1e-11`（同一批数字、不同的浮点求和顺序所致），选出的超参数完全相同；
- 所有排序逐位一致，**唯一例外**是文档在 `top_k` 截断处**并列同分**时：`np.argpartition` 会从并列分组里任取一个，而 Rust 采用 Python 代码自己写明的 tie-break（文档下标小者优先）。两者分数逐位相同，MRR@10 与 NDCG@10 不受影响；当第 100 名那个并列文档恰好在其中一边被判为相关时，Recall@100 会有第 4 位小数级别的差异。

## 局限

- **分布外（OOD）**：每个条目是在它自己的来源语料上验证的，不是你手上的语料。把目标语料自己的
  条目排除后，字典在上表 16 个语料上仍有 75% 命中率——但此时它是在"拿别的语料"做预测，这不是
  免费的：留一语料（leave-one-out）设置下 12/16 有提升（均值 +17.0%，中位数 +7.7%），
  4 个低于默认参数（-8% 到 -23%：fiqa、arguana、climate-fever、vihealthqa，其中三个是字典
  未命中、直接落到启发式）。OOD 防护和启发式兜底能限制损失，但消除不了它。如果你的语料在字典里
  没有对应物，请先验证它给出的参数再上线。
- **非英文语料**：有少数特征是语言相关的（英文停用词密度、去停用词后的词频统计）。越南语
  vihealthqa 已经进了字典（+2.2%），但换一门字典没见过的语言，建议先验证返回的参数再上线。

## 项目结构

```
autobm25/
├── autobm25/             # pip 包：检索接口（引擎、特征、词典、CLI）+ data/
├── rust/                 # Rust 内核 + PyO3 绑定（python rust/build_pyext.py）
├── tests/                # pytest：API、加载器、指标、词典与双后端一致性
├── .github/workflows/    # 三平台 CI 与 Trusted Publishing
├── examples/             # 可直接运行的库用法示例（quickstart.py 等）
├── logo/
├── config.yaml           # 启发式系数
├── setup.py              # 把编译好的扩展打进 wheel（元数据在 pyproject.toml）
├── pyproject.toml        # pip 打包配置
├── requirements.txt
└── README.md / README_CN.md
```

## License

本项目采用 [MIT License](LICENSE)：代码可**自由复制、修改、商用**，唯一要求是在副本中**保留版权声明与本许可文本**（即"使用请注明出处"）。引用本项目时附上仓库链接即可满足要求。

---

🇬🇧 English version: [README.md](README.md)
