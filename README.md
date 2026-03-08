\![Visitors](https://visitor-badge.laobi.icu/badge?page_id=24kchengYe.Urban-Renewal-Simulation-based-on-Agent-and-Graph-Systems)
# GPGLB: Government and Resident Participation Graph Lingual Network for Buildings

## 政府和居民参与的建筑图语言网络模型

一个结合**图神经网络(GNN)**和**大语言模型(LLM)**的建筑质量演化模拟系统，用于城市更新决策支持。

---

## 🎯 项目概述

### 核心创新点

1. **LLM作为图神经消息传递单元**：节点间传递自然语言文本而非高维向量
2. **双层Agent决策系统**：政府Agent制定政策 + 居民Agent响应政策
3. **真实数据驱动**：基于西宁市2022-2025年建筑质量检测真值
4. **可解释性强**：每个建筑质量变化都有LLM生成的推理过程

### 理论基础

- **破窗理论**：周围建筑质量影响目标建筑的维护决策
- **空间溢出效应**：邻域环境通过视觉、经济、社会机制传递
- **双层决策博弈**：政府(上层)政策激励 ↔ 居民(下层)投资决策
- **时空图网络**：建筑质量在时空图上的游走演化

---

## 📁 项目结构

```
AI_ABM_builidngscalesimulation/
├── README.md                      # 本文档
├── requirements.txt               # Python依赖包
├── .env.example                   # API配置模板
├── .gitignore                     # Git忽略文件
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   │   ├── buildings_2022.shp     # 2022年建筑矢量
│   │   ├── buildings_2023.shp     # 2023年建筑矢量
│   │   ├── buildings_2024.shp     # 2024年建筑矢量
│   │   └── buildings_2025.shp     # 2025年建筑矢量
│   ├── processed/                 # 处理后的图数据
│   │   ├── graph_2022.pkl         # 2022年建筑图
│   │   ├── graph_2023.pkl         # 2023年建筑图
│   │   └── node_features.csv      # 节点特征表
│   └── external/                  # 外部数据
│       ├── poi_xining.csv         # POI数据
│       ├── pedestrian_flow.csv    # 人流量数据
│       └── metro_stations.geojson # 地铁站点
│
├── models/                        # 模型代码
│   ├── __init__.py
│   ├── gnn_baseline.py           # 传统GNN基线模型
│   ├── llm_gln.py                # LLM图语言网络
│   ├── building_agent.py         # 建筑物业主Agent
│   └── government_agent.py       # 政府Agent
│
├── experiments/                   # 实验脚本
│   ├── 01_train_gnn_baseline.py  # 训练GNN基线
│   ├── 02_calibrate_llm.py       # 校准LLM-GLN
│   ├── 03_policy_simulation.py   # 政策场景模拟
│   └── 04_comparative_analysis.py # 对比分析
│
├── visualization/                 # 可视化工具
│   ├── __init__.py
│   ├── spatial_map.py            # 空间地图可视化
│   ├── quality_evolution.py      # 质量演化动画
│   └── metric_dashboard.py       # 指标仪表盘
│
├── utils/                         # 工具函数
│   ├── __init__.py
│   ├── graph_builder.py          # 图构建工具
│   ├── data_loader.py            # 数据加载器
│   └── metrics.py                # 评估指标
│
├── notebooks/                     # Jupyter笔记本
│   ├── 01_data_exploration.ipynb # 数据探索
│   ├── 02_gnn_experiments.ipynb  # GNN实验
│   └── 03_llm_analysis.ipynb     # LLM分析
│
└── config/                        # 配置文件
    ├── gnn_config.yaml           # GNN配置
    └── llm_config.yaml           # LLM配置
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目（或直接打开文件夹）
cd AI_ABM_builidngscalesimulation

# 安装依赖
pip install -r requirements.txt

# 配置API密钥
cp .env.example .env
# 编辑 .env 文件，填入OpenAI API密钥
```

### 2. 数据准备

```bash
# 将西宁市建筑Shp文件放入 data/raw/ 目录
# 运行数据预处理
python utils/data_loader.py
```

### 3. 运行实验

#### 阶段1: GNN基线模型

```bash
# 训练传统图神经网络
python experiments/01_train_gnn_baseline.py

# 输出：
# - models/checkpoints/gnn_best.pth
# - results/gnn_baseline_metrics.json
```

#### 阶段2: LLM-GLN校准

```bash
# 使用2022→2023真值校准LLM
python experiments/02_calibrate_llm.py

# 输出：
# - config/llm_calibrated_prompt.txt
# - results/llm_calibration_log.json
```

#### 阶段3: 政策场景模拟

```bash
# 模拟2025→2030三种政策场景
python experiments/03_policy_simulation.py --scenario all

# 场景选项:
#   --scenario trend      # 趋势发展
#   --scenario cbd        # CBD导向
#   --scenario tod        # TOD导向
#   --scenario all        # 全部场景
```

#### 阶段4: 对比分析

```bash
# 生成对比报告和可视化
python experiments/04_comparative_analysis.py
```

---

## 📊 核心模型架构

### 1. 传统GNN基线 (models/gnn_baseline.py)

```python
# 时空图卷积网络
class BuildingQualityGNN(nn.Module):
    """
    输入: 节点特征X ∈ R^{N×F}, 邻接矩阵A
    输出: 质量变化 Δquality ∈ [-2, 2]

    架构:
    - 3层GAT (Graph Attention Network)
    - LSTM时序建模
    - 全连接预测层
    """
```

### 2. LLM-GLN模型 (models/llm_gln.py)

```python
# 图语言网络消息传递
async def message_passing_round(G, llm_client):
    """
    每个节点:
    1. 聚合邻居信息 → 自然语言描述
    2. LLM推理 → 输出决策和质量变化
    3. 更新图状态

    Prompt结构:
    - 建筑当前状态
    - 周边环境描述
    - 业主个性和记忆
    - 政策激励信息
    - 决策任务
    """
```

### 3. 双层Agent系统

#### 居民Agent (models/building_agent.py)

```python
class BuildingOwnerAgent:
    """
    业主类型:
    - resident: 自住业主（保守，注重舒适）
    - investor: 投资者（激进，追求收益）
    - commercial: 商业业主（平衡，关注人流）

    决策逻辑:
    - 感知邻域质量 + POI密度 + 客流量
    - 考虑历史维护记录和预算
    - 响应政府政策激励
    """
```

#### 政府Agent (models/government_agent.py)

```python
class GovernmentAgent:
    """
    政策类型:
    1. 趋势发展: 无干预
    2. CBD导向: 商业区财政补贴
    3. TOD导向: 地铁站周边容积率奖励

    政策工具:
    - 财政补贴
    - 容积率奖励
    - 税收减免
    - 快速审批
    """
```

---

## 🧪 实验设计

### 实验1: 模型性能对比

| 模型 | MAE | RMSE | Accuracy | 可解释性 |
|------|-----|------|----------|---------|
| GNN基线 | ? | ? | ? | ❌ 黑箱 |
| LLM-GLN | ? | ? | ? | ✅ 自然语言推理 |
| ABM | ? | ? | ? | ⚠️ 规则可见 |
| CA-ABM | ? | ? | ? | ⚠️ 规则可见 |

### 实验2: 校准过程

```
2022年真值 → 模拟2023 → 与2023真值对比 → 调整Prompt
重复直至MAE最小
```

### 实验3: 政策场景对比

| 场景 | 高质量建筑比例 | 空间聚集度 | 财政投入 |
|------|---------------|-----------|---------|
| 趋势发展 | ? | ? | 0元 |
| CBD导向 | ? | ? | ?万元 |
| TOD导向 | ? | ? | ?万元 |

---

## 📈 可视化输出

### 1. 空间地图 (visualization/spatial_map.py)

```python
# 交互式地图展示建筑质量分布
create_folium_map(G, year=2025)

# 输出：results/figures/quality_map_2025.html
```

### 2. 演化动画 (visualization/quality_evolution.py)

```python
# 生成2022→2030质量演化GIF
create_evolution_animation(
    graphs=[G_2022, G_2023, ..., G_2030],
    output='results/figures/evolution.gif'
)
```

### 3. 决策热力图

```python
# 可视化不同政策下的翻新决策分布
plot_renovation_heatmap(scenario='CBD导向')
```

---

## ⚙️ 配置参数

### GNN配置 (config/gnn_config.yaml)

```yaml
model:
  hidden_dim: 64
  num_layers: 3
  heads: 4
  dropout: 0.3

training:
  epochs: 200
  learning_rate: 0.001
  batch_size: 512
  early_stopping: 20
```

### LLM配置 (config/llm_config.yaml)

```yaml
api:
  model: "gpt-4o-mini"
  temperature: 0.3
  max_tokens: 200
  concurrency_limit: 20

simulation:
  llm_trigger_prob: 1.0  # 所有节点都用LLM
  save_reasoning: true
  cache_similar_prompts: true
```

---

## 📚 依赖包

详见 `requirements.txt`:

```txt
# 空间数据处理
geopandas>=0.12.0
shapely>=2.0.0
osmnx>=1.3.0
momepy>=0.6.0

# 图神经网络
torch>=2.0.0
torch-geometric>=2.3.0

# LLM调用
openai>=1.0.0

# 数据科学
numpy>=1.24.0
pandas>=1.5.0
scipy>=1.10.0

# 可视化
matplotlib>=3.7.0
folium>=0.14.0
plotly>=5.14.0

# 空间统计
pysal>=23.1
```

---

## 🔬 研究问题

### RQ1: LLM vs GNN在建筑质量预测上的性能差异？

**假设**: LLM能捕捉GNN忽略的主观因素（业主心理、文化偏好）

### RQ2: 双层Agent系统能否更准确模拟政策效果？

**假设**: 显式建模政府-居民博弈比单层模型更接近真实

### RQ3: 可解释性是否牺牲了预测精度？

**假设**: LLM虽可解释，但可能不如深度学习精确

### RQ4: 不同政策场景下的空间涌现模式？

**假设**: CBD政策导致中心集聚，TOD政策导致多中心格局

---

## 🛠️ 开发路线图

### Phase 1: 数据准备（已完成）
- [x] 西宁市建筑Shp数据获取
- [x] 质量真值对齐
- [ ] POI数据爬取
- [ ] 街道网络下载

### Phase 2: GNN基线（2周）
- [ ] 实现ST-GCN模型
- [ ] 2022→2023训练
- [ ] 2023→2024验证

### Phase 3: LLM-GLN（3周）
- [ ] Agent类设计
- [ ] Prompt工程
- [ ] 消息传递实现
- [ ] 校准流程

### Phase 4: 双层决策（2周）
- [ ] 政府Agent集成
- [ ] 三种场景实现
- [ ] 2025→2030模拟

### Phase 5: 分析报告（1周）
- [ ] 指标对比
- [ ] 可视化生成
- [ ] 论文撰写

---

## 📖 参考文献

1. **破窗理论**: Wilson & Kelling (1982) "Broken Windows"
2. **空间溢出**: Tobler's First Law of Geography
3. **图神经网络**: Kipf & Welling (2017) "Semi-Supervised Classification with GCN"
4. **LLM-Agent**: Park et al. (2023) "Generative Agents: Interactive Simulacra"
5. **城市更新**: Zheng et al. (2024) "Urban Renewal Dynamics"

---

## 📧 联系方式

项目维护者: [你的名字]
邮箱: [your.email@example.com]

---

## 📄 许可证

本项目仅供学术研究使用。

## 更新日志

### 2026-01-16
- 项目初始化
- 创建目录结构
- 编写技术设计文档
