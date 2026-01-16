"""
==============================================================================
LLM图语言网络 (LLM Graph Lingual Network)
==============================================================================
核心创新：使用LLM作为图神经网络的消息传递单元

传统GNN: 节点间传递高维数学向量
LLM-GLN: 节点间传递自然语言文本

消息传递流程：
1. 聚合邻域信息 → 自然语言描述
2. 构造Prompt（建筑状态 + 邻域 + 政策）
3. LLM推理 → 输出决策和质量变化
4. 更新图状态

优势：
- 可解释性：每个决策都有自然语言推理过程
- 灵活性：可轻松融合非结构化信息（政策文本、历史记忆）
- 主观性：捕捉难以量化的主观因素（业主心理、文化偏好）
==============================================================================
"""

import os
import asyncio
import json
import networkx as nx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm
from typing import Dict, List, Tuple
import httpx

from .building_agent import BuildingOwnerAgent
from .government_agent import GovernmentAgent


class LLM_GLN_Simulator:
    """LLM图语言网络模拟器"""

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = "gpt-4o-mini", temperature: float = 0.3,
                 concurrency_limit: int = 20):
        """
        初始化模拟器

        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型
            temperature: 温度参数
            concurrency_limit: 并发请求数限制
        """
        # 加载环境变量
        load_dotenv()

        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.model = model
        self.temperature = temperature

        # 创建异步客户端
        proxy = os.getenv('https_proxy') or os.getenv('HTTPS_PROXY')
        if proxy:
            http_client = httpx.AsyncClient(proxy=proxy, timeout=60.0)
        else:
            http_client = httpx.AsyncClient(timeout=60.0)

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client
        )

        # 并发控制
        self.semaphore = asyncio.Semaphore(concurrency_limit)

        # 统计信息
        self.stats = {
            'total_api_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_cost': 0.0
        }

        print(f"🤖 LLM-GLN模拟器初始化完成")
        print(f"   模型: {self.model}")
        print(f"   并发限制: {concurrency_limit}")

    async def call_llm(self, prompt: str) -> str:
        """
        调用LLM API

        Args:
            prompt: 输入提示词

        Returns:
            response: LLM返回的文本
        """
        async with self.semaphore:
            try:
                self.stats['total_api_calls'] += 1

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                self.stats['successful_calls'] += 1

                # 估算成本（简化版）
                self.stats['total_cost'] += 0.0001  # gpt-4o-mini约$0.15/1M tokens

                return content

            except Exception as e:
                self.stats['failed_calls'] += 1
                print(f"⚠️ LLM调用失败: {e}")
                return json.dumps({
                    "decision": "维持现状",
                    "expected_quality_change": -1,
                    "estimated_cost": 0,
                    "reasoning": f"API调用失败: {str(e)[:50]}"
                })

    async def process_building(self, node_id: int, G: nx.Graph,
                               government_agent: GovernmentAgent,
                               current_year: int) -> Dict:
        """
        处理单个建筑的决策

        Args:
            node_id: 建筑节点ID
            G: 建筑图
            government_agent: 政府Agent
            current_year: 当前年份

        Returns:
            decision: 决策结果
        """
        # 获取建筑Agent
        agent = G.nodes[node_id]['agent']

        # 1. 收集邻域信息
        neighbors = list(G.neighbors(node_id))
        neighbor_data = [
            {
                'id': n,
                'quality': G.nodes[n].get('quality', 3),
                'distance': G[node_id][n]['distance']
            }
            for n in neighbors
        ]

        # 2. 收集外部特征
        external_features = {
            'poi_density_500m': G.nodes[node_id].get('poi_density_500m', 0),
            'pedestrian_flow': G.nodes[node_id].get('pedestrian_flow', 0),
            'building_density': G.nodes[node_id].get('building_density', 0),
            'distance_to_cbd': G.nodes[node_id].get('distance_to_cbd', 0),
            'distance_to_metro': G.nodes[node_id].get('distance_to_metro', 9999)
        }

        # 3. 感知环境
        perception = agent.perceive_environment(neighbor_data, external_features)

        # 4. 生成政策上下文
        policy_context = government_agent.generate_policy_context(G.nodes[node_id])

        # 5. 生成Prompt
        prompt = agent.generate_decision_context(perception, current_year, policy_context)

        # 6. 调用LLM
        llm_output = await self.call_llm(prompt)

        # 7. 解析决策
        decision = agent.parse_llm_response(llm_output)
        decision['node_id'] = node_id

        return decision

    async def simulate_one_year(self, G: nx.Graph,
                                government_agent: GovernmentAgent,
                                current_year: int) -> nx.Graph:
        """
        模拟一年的演化

        Args:
            G: 建筑图
            government_agent: 政府Agent
            current_year: 当前年份

        Returns:
            G: 更新后的图
        """
        print(f"\n{'='*60}")
        print(f"🏙️ 模拟 {current_year} → {current_year+1}")
        print(f"{'='*60}")

        # 创建所有建筑的异步任务
        tasks = [
            self.process_building(nid, G, government_agent, current_year)
            for nid in G.nodes()
        ]

        # 并发执行，使用tqdm显示进度
        print(f"📡 调用LLM进行决策推理（共{len(tasks)}个建筑）...")
        decisions = await async_tqdm.gather(*tasks, desc="   LLM API调用")

        # 统计决策分布
        decision_counts = {}
        quality_changes = []
        total_cost = 0
        total_subsidy = 0

        for decision in decisions:
            node_id = decision['node_id']
            agent = G.nodes[node_id]['agent']

            # 统计决策类型
            dec_type = decision['decision']
            decision_counts[dec_type] = decision_counts.get(dec_type, 0) + 1

            # 应用政府补贴
            subsidy = government_agent.apply_subsidy(node_id, decision)
            total_subsidy += subsidy

            # 计算实际成本
            actual_cost = decision['estimated_cost'] - subsidy
            total_cost += actual_cost

            # 更新Agent状态
            agent.update_state(decision, current_year)

            # 更新图节点质量
            G.nodes[node_id]['quality'] = agent.state['quality']
            G.nodes[node_id]['last_decision'] = decision['decision']
            G.nodes[node_id]['reasoning'] = decision['reasoning']

            quality_changes.append(decision['expected_quality_change'])

        # 打印统计信息
        print(f"\n📊 本年度统计:")
        print(f"   决策分布:")
        for dec_type, count in sorted(decision_counts.items()):
            print(f"      {dec_type}: {count}栋 ({count/len(decisions)*100:.1f}%)")

        print(f"\n   质量变化:")
        print(f"      平均变化: {sum(quality_changes)/len(quality_changes):+.2f}")
        print(f"      改善建筑: {sum(1 for q in quality_changes if q > 0)}栋")
        print(f"      恶化建筑: {sum(1 for q in quality_changes if q < 0)}栋")

        print(f"\n   经济成本:")
        print(f"      总成本: {total_cost:.1f}万元")
        print(f"      政府补贴: {total_subsidy:.1f}万元")
        print(f"      业主实际支出: {total_cost - total_subsidy:.1f}万元")

        print(f"\n   API统计:")
        print(f"      成功率: {self.stats['successful_calls']/self.stats['total_api_calls']*100:.1f}%")
        print(f"      累计成本: ${self.stats['total_cost']:.2f}")

        return G

    async def run_simulation(self, G_initial: nx.Graph,
                            government_agent: GovernmentAgent,
                            start_year: int, end_year: int,
                            save_interval: int = 1,
                            save_dir: str = "./results") -> Dict[int, nx.Graph]:
        """
        运行完整的多年模拟

        Args:
            G_initial: 初始建筑图
            government_agent: 政府Agent
            start_year: 起始年份
            end_year: 结束年份
            save_interval: 保存间隔（年）
            save_dir: 保存目录

        Returns:
            graphs_history: {year: Graph}
        """
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║            LLM-GLN 城市建筑质量演化模拟                       ║
║            {start_year} → {end_year} ({end_year - start_year}年)                                 ║
╚══════════════════════════════════════════════════════════════╝
        """)

        G = G_initial.copy()
        graphs_history = {start_year: G.copy()}
        current_year = start_year

        # 初始化所有建筑的Agent
        print(f"\n🏗️ 初始化建筑Agent...")
        owner_types = self._infer_owner_types(G)

        for idx, (nid, owner_type) in enumerate(owner_types.items()):
            building_data = dict(G.nodes[nid])
            building_data['building_id'] = nid
            G.nodes[nid]['agent'] = BuildingOwnerAgent(
                nid, building_data, owner_type, random_seed=42
            )

        print(f"   总计 {len(G.nodes())} 个建筑Agent")
        print(f"   业主类型分布: {self._count_owner_types(G)}")

        # 逐年模拟
        while current_year < end_year:
            G = await self.simulate_one_year(G, government_agent, current_year)
            current_year += 1

            # 保存中间结果
            if current_year % save_interval == 0:
                graphs_history[current_year] = G.copy()
                self._save_graph(G, current_year, save_dir)

        # 保存最终结果
        graphs_history[current_year] = G.copy()
        self._save_graph(G, current_year, save_dir)

        print(f"\n{'='*60}")
        print(f"✅ 模拟完成! ({start_year} → {end_year})")
        print(f"{'='*60}")

        return graphs_history

    def _infer_owner_types(self, G: nx.Graph) -> Dict[int, str]:
        """
        根据建筑属性推断业主类型

        Returns:
            owner_types: {node_id: owner_type}
        """
        import numpy as np
        owner_types = {}

        for nid in G.nodes():
            use_type = G.nodes[nid].get('use_type', '未知')
            area = G.nodes[nid].get('area', 100)

            if use_type == '商业':
                owner_types[nid] = 'commercial'
            elif use_type == '住宅':
                if area > 200:
                    # 大户型，30%概率是投资者
                    owner_types[nid] = np.random.choice(
                        ['resident', 'investor'],
                        p=[0.7, 0.3]
                    )
                else:
                    owner_types[nid] = 'resident'
            else:
                owner_types[nid] = 'resident'

        return owner_types

    def _count_owner_types(self, G: nx.Graph) -> Dict[str, int]:
        """统计业主类型分布"""
        counts = {'resident': 0, 'investor': 0, 'commercial': 0}
        for nid in G.nodes():
            owner_type = G.nodes[nid]['agent'].owner_type
            counts[owner_type] += 1
        return counts

    def _save_graph(self, G: nx.Graph, year: int, save_dir: str):
        """保存图快照"""
        import pickle
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, f"graph_{year}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(G, f)

        print(f"   💾 已保存: graph_{year}.pkl")


# ==============================================================================
# 使用示例
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append('..')

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              LLM-GLN模拟器 使用示例                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print("\n提示：完整示例请参考 experiments/02_calibrate_llm.py")
    print("      这里展示基本用法：")

    # 示例：模拟单年
    async def demo():
        # 假设你已经有建筑图G和政府Agent
        # from utils.graph_builder import GraphBuilder
        # builder = GraphBuilder('visibility')
        # G = builder.build_graph(buildings_gdf)

        # gov_agent = GovernmentAgent('cbd', budget=1e8)

        # simulator = LLM_GLN_Simulator(concurrency_limit=10)
        # G_2023 = await simulator.simulate_one_year(G_2022, gov_agent, 2022)

        print("\n（示例代码，需要真实数据才能运行）")

    # asyncio.run(demo())
