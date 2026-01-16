# GPGLB 快速启动指南

本文档帮助你在5分钟内运行GPGLB项目的第一个模拟。

---

## ⚡ 5分钟快速启动

### 步骤1: 安装依赖（2分钟）

```bash
# 进入项目目录
cd AI_ABM_builidngscalesimulation

# 安装Python依赖
pip install -r requirements.txt

# 如果速度慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤2: 配置API密钥（1分钟）

```bash
# 复制配置模板
copy .env.example .env  # Windows
# 或
cp .env.example .env    # Linux/Mac

# 编辑 .env 文件
# 1. 找到 OPENAI_API_KEY=your_api_key_here
# 2. 替换为你的实际API密钥
# 3. 保存文件
```

**获取API密钥：**
- [OpenAI官网](https://platform.openai.com/api-keys)
- [编协AI](https://bianxie.ai/)（国内可用）
- [DeepSeek](https://platform.deepseek.com/)

### 步骤3: 运行演示（2分钟）

```bash
# 运行端到端演示脚本
python experiments/demo_end_to_end.py
```

**第一次运行会自动：**
1. 创建500个模拟建筑（因为你还没有真实数据）
2. 构建视线可达性图
3. 运行1年的LLM-GLN模拟（2022→2023）
4. 生成分析报告

**预期输出：**
```
╔══════════════════════════════════════════════════════════════╗
║     GPGLB: Government and Resident Participation             ║
║            Graph Lingual Network for Buildings               ║
╚══════════════════════════════════════════════════════════════╝

阶段1: 数据加载与预处理
✅ 创建了 500 个模拟建筑

阶段2: 构建建筑空间图网络
🏗️ 开始构建建筑图 (类型: visibility)...
✅ 图构建完成!
   节点数: 500
   边数: 2134
   平均度: 8.54

阶段3: LLM-GLN模拟
🏛️ 创建政府Agent (场景: cbd, 预算: 1.0亿元)
📡 调用LLM进行决策推理（共500个建筑）...
   LLM API调用: 100%|████████████| 500/500

📊 本年度统计:
   决策分布:
      一般维护: 180栋 (36.0%)
      基础维修: 150栋 (30.0%)
      维持现状: 120栋 (24.0%)
      重大翻新: 50栋 (10.0%)

阶段4: 结果分析与可视化
╔══════════════════════════════════════════════════════════════╗
║             政策影响评估报告                                  ║
║             场景: cbd                                         ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📚 下一步

### 使用真实数据

将你的西宁市建筑Shapefile放入`data/raw/`目录：

```
data/raw/
├── buildings_2022.shp  # 2022年建筑数据
├── buildings_2023.shp  # 2023年建筑数据
├── buildings_2024.shp  # 2024年建筑数据
└── buildings_2025.shp  # 2025年建筑数据
```

**必需字段：**
- `quality`: 建筑质量等级 [1-5]
- `year_built`: 建成年份
- `area`: 建筑面积（㎡）
- `floors`: 楼层数
- `use_type`: 用途（"住宅"/"商业"/"混合"）

然后修改`experiments/demo_end_to_end.py`中的路径。

### 运行完整实验

```bash
# 1. 训练GNN基线模型
python experiments/01_train_gnn_baseline.py

# 2. 校准LLM-GLN（2022→2023）
python experiments/02_calibrate_llm.py

# 3. 政策场景模拟（2025→2030）
python experiments/03_policy_simulation.py --scenario all

# 4. 生成对比分析报告
python experiments/04_comparative_analysis.py
```

### 调整参数

编辑`.env`文件调整关键参数：

```env
# 降低成本（使用更便宜的模型）
OPENAI_MODEL=gpt-4o-mini

# 提高速度（降低LLM触发概率）
LLM_TRIGGER_PROB=0.3

# 减少规模（测试用）
GRAPH_TYPE=euclidean
K_NEIGHBORS=4
MAX_DISTANCE=200
```

---

## 🐛 常见问题

### Q1: 安装torch-geometric失败

**A:** PyTorch Geometric需要匹配PyTorch版本：

```bash
# 先安装PyTorch
pip install torch torchvision torchaudio

# 再按官方指南安装PyG
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

### Q2: osmnx安装报错（Windows）

**A:** Windows用户推荐使用conda：

```bash
conda install -c conda-forge osmnx
```

或跳过osmnx（只使用欧氏距离图）：

```env
# .env文件
GRAPH_TYPE=euclidean  # 不使用街道网络
```

### Q3: API调用失败（429 Too Many Requests）

**A:** 降低并发数：

```env
# .env文件
API_CONCURRENCY_LIMIT=5  # 从20降到5
```

### Q4: 内存不足

**A:** 减小规模或分批处理：

```env
# .env文件
BATCH_PROCESSING=true
BATCH_SIZE_NODES=500
```

### Q5: 模拟太慢

**A:** 优化策略：

```env
# 1. 使用更快的模型
OPENAI_MODEL=gpt-4o-mini  # 或 google/gemini-flash-1.5

# 2. 降低LLM调用频率
LLM_TRIGGER_PROB=0.3

# 3. 增加并发
API_CONCURRENCY_LIMIT=20

# 4. 减少建筑数量（测试阶段）
# 在demo脚本中修改：create_mock_buildings(n=100)
```

---

## 📖 学习路径

### 初学者

1. ✅ 运行`demo_end_to_end.py`了解整体流程
2. 📚 阅读`README.md`理解理论基础
3. 🔍 查看`models/building_agent.py`了解Agent设计
4. 🎨 运行`visualization/`中的脚本生成图表

### 进阶用户

1. 🧪 修改`config/llm_config.yaml`调整Prompt
2. 🏗️ 自定义`utils/graph_builder.py`构建新图类型
3. 🤖 扩展`models/government_agent.py`添加新政策
4. 📊 实现自己的评估指标

### 研究者

1. 📝 对比GNN vs LLM-GLN的性能差异
2. 🔬 设计消融实验（去除某个特征观察影响）
3. 🌍 应用到其他城市数据
4. 📄 撰写学术论文

---

## 💡 提示

### 节省API成本

```python
# 在demo脚本中，先用少量建筑测试
buildings_gdf = create_mock_buildings(n=50)  # 只用50个建筑

# 确认流程正确后，再用完整数据
buildings_gdf = create_mock_buildings(n=500)
```

### 调试模式

```env
# .env文件
LOG_LEVEL=DEBUG  # 输出详细日志
SAVE_REASONING=true  # 保存LLM推理过程
```

### 性能监控

运行后检查`results/logs/`目录：
- `api_calls.log`: API调用记录
- `performance.log`: 性能统计

---

## 🆘 获取帮助

- 📧 邮件: [your.email@example.com]
- 💬 GitHub Issues: [项目地址]/issues
- 📖 完整文档: 见`README.md`

---

**祝你使用愉快！🎉**

如果这个项目对你有帮助，请给我们一个⭐Star！
