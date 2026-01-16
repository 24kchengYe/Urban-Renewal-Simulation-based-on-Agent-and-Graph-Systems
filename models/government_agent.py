"""
==============================================================================
政府智能体 (Government Agent)
==============================================================================
功能：模拟市政府在城市更新中的政策制定行为

政策类型：
1. 趋势发展场景：无政府干预，市场自然演化
2. CBD导向场景：鼓励商业中心区域的建筑翻新（财政补贴）
3. TOD导向场景：鼓励地铁站周边的建筑升级（容积率奖励）

政策工具：
- 财政补贴（降低业主成本）
- 容积率奖励（增加开发收益）
- 税收减免（降低运营成本）
- 快速审批（降低时间成本）
==============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple


class GovernmentAgent:
    """市政府智能体"""

    def __init__(self, policy_scenario: str, budget: float = 1e8,
                 city_center_coords: Tuple[float, float] = None):
        """
        初始化政府Agent

        Args:
            policy_scenario: 政策场景 ['trend', 'cbd', 'tod']
            budget: 财政预算（元）
            city_center_coords: 城市中心坐标 (lon, lat)
        """
        self.policy_scenario = policy_scenario
        self.total_budget = budget
        self.remaining_budget = budget
        self.city_center = city_center_coords

        # 政策执行记录
        self.policy_log = {
            'subsidies_granted': [],
            'buildings_affected': [],
            'total_expenditure': 0
        }

        print(f"🏛️ 政府Agent初始化 (场景: {policy_scenario}, 预算: {budget/1e8:.1f}亿元)")

    def generate_policy_context(self, building_node: Dict) -> str:
        """
        根据政策场景生成针对性的政策描述

        Args:
            building_node: 建筑节点数据（包含质量、位置等属性）

        Returns:
            policy_text: 插入到业主Prompt中的政策描述
        """
        if self.policy_scenario == 'trend':
            # 趋势发展：无政府干预
            return "无特殊的财政补贴或容积率激励政策。"

        elif self.policy_scenario == 'cbd':
            # CBD导向：商业中心区域激励
            return self._generate_cbd_policy(building_node)

        elif self.policy_scenario == 'tod':
            # TOD导向：地铁站周边激励
            return self._generate_tod_policy(building_node)

        else:
            return "政策正在制定中。"

    def _generate_cbd_policy(self, building_node: Dict) -> str:
        """
        生成CBD导向政策描述

        条件：距CBD < 2km 且 质量 >= 3（一般/较差/危房）
        """
        dist_to_cbd = building_node.get('distance_to_cbd', float('inf'))
        quality = building_node.get('quality', 3)
        building_id = building_node.get('building_id', 0)

        # 判断是否符合补贴条件
        if dist_to_cbd < 2000 and quality >= 3:
            # 符合条件
            subsidy_rate = 0.5  # 50%补贴
            max_subsidy = 30  # 最高30万

            # 检查预算是否充足
            if self.remaining_budget >= max_subsidy * 10000:
                policy_text = f"""
[新激励政策 - CBD更新计划]
市政府为鼓励CBD区域的商业活力，现为您的建筑翻新提供以下支持：

✅ 您的建筑符合补贴条件（距CBD {dist_to_cbd:.0f}米）

政策工具：
1. 财政补贴：翻新成本的{subsidy_rate*100:.0f}%（最高{max_subsidy}万元）
2. 容积率奖励：可增加0.5倍建筑面积
3. 税收减免：翻新后前3年免征房产税
4. 快速审批：优先审批，15个工作日内完成

适用范围：距CBD {2000}米内，质量等级3级及以下的建筑
有效期：2025-2030年
"""
                # 预留补贴预算
                self._record_potential_subsidy(building_id, max_subsidy)
                return policy_text

        # 不符合条件或预算不足
        return f"""
[CBD更新计划]
您的建筑不在本轮CBD更新激励范围内。

条件：距CBD < 2km 且 质量等级 >= 3
您的建筑：距CBD {dist_to_cbd:.0f}米，质量等级 {quality}
"""

    def _generate_tod_policy(self, building_node: Dict) -> str:
        """
        生成TOD导向政策描述

        条件：距地铁站 < 800m 且 质量 >= 3
        """
        dist_to_metro = building_node.get('distance_to_metro', float('inf'))
        quality = building_node.get('quality', 3)
        building_id = building_node.get('building_id', 0)
        use_type = building_node.get('use_type', '未知')

        # 判断是否符合激励条件
        if dist_to_metro < 800 and quality >= 3:
            # 容积率奖励力度（住宅更高）
            far_bonus = 1.0 if use_type == '住宅' else 0.5

            if self.remaining_budget >= 20 * 10000:  # 预留20万预算
                policy_text = f"""
[新激励政策 - TOD绿色发展]
市政府为推动TOD（交通导向开发）发展，现为地铁站周边建筑升级提供：

✅ 您的建筑符合TOD激励条件（距地铁站 {dist_to_metro:.0f}米）

政策工具：
1. 容积率奖励：可增加{far_bonus}倍建筑面积（{'住宅优先' if use_type == '住宅' else '商业适用'}）
2. 绿色建筑认证：满足二星级标准可获20万元补贴
3. 快速审批通道：审批时间缩短至1个月
4. 地铁站联动开发：优先考虑与地铁站商业空间联合开发

适用范围：地铁站{800}米范围内，质量等级3级及以下
有效期：2025-2030年
"""
                self._record_potential_subsidy(building_id, 20)
                return policy_text

        # 不符合条件
        return f"""
[TOD绿色发展计划]
您的建筑不在本轮TOD激励范围内。

条件：距地铁站 < 800米 且 质量等级 >= 3
您的建筑：距地铁站 {dist_to_metro:.0f}米，质量等级 {quality}
"""

    def _record_potential_subsidy(self, building_id: int, amount: float):
        """
        记录潜在的补贴支出

        Args:
            building_id: 建筑ID
            amount: 补贴金额（万元）
        """
        self.policy_log['buildings_affected'].append(building_id)
        # 注意：实际支出在业主决策后才确定

    def apply_subsidy(self, building_id: int, decision: Dict) -> float:
        """
        应用财政补贴（在业主做出决策后调用）

        Args:
            building_id: 建筑ID
            decision: 业主决策（包含estimated_cost）

        Returns:
            actual_subsidy: 实际补贴金额（万元）
        """
        if self.policy_scenario == 'trend':
            return 0.0  # 无补贴

        # 只有重大翻新和一般维护才有补贴
        if decision['decision'] not in ['重大翻新', '一般维护']:
            return 0.0

        cost = decision.get('estimated_cost', 0)

        # CBD场景：50%补贴，上限30万
        if self.policy_scenario == 'cbd':
            subsidy = min(cost * 0.5, 30)

        # TOD场景：绿色建筑认证补贴20万（简化为固定值）
        elif self.policy_scenario == 'tod':
            subsidy = 20 if decision['decision'] == '重大翻新' else 10

        else:
            subsidy = 0.0

        # 检查预算约束
        if self.remaining_budget >= subsidy * 10000:
            self.remaining_budget -= subsidy * 10000
            self.policy_log['total_expenditure'] += subsidy * 10000
            self.policy_log['subsidies_granted'].append({
                'building_id': building_id,
                'amount': subsidy,
                'decision': decision['decision']
            })
            return subsidy
        else:
            print(f"⚠️ 财政预算不足，无法为建筑{building_id}提供补贴")
            return 0.0

    def evaluate_policy_impact(self, G_before, G_after) -> Dict:
        """
        评估政策效果

        Args:
            G_before: 政策实施前的图
            G_after: 政策实施后的图

        Returns:
            metrics: 评估指标字典
        """
        print(f"\n📊 评估政策影响 (场景: {self.policy_scenario})")

        # 1. 质量改善统计
        improved_buildings = []
        deteriorated_buildings = []

        for nid in G_before.nodes():
            quality_before = G_before.nodes[nid].get('quality', 3)
            quality_after = G_after.nodes[nid].get('quality', 3)

            if quality_after < quality_before:
                improved_buildings.append(nid)
            elif quality_after > quality_before:
                deteriorated_buildings.append(nid)

        # 2. 质量分布变化
        qualities_before = [G_before.nodes[n]['quality'] for n in G_before.nodes()]
        qualities_after = [G_after.nodes[n]['quality'] for n in G_after.nodes()]

        # 3. 空间聚集度（简化版：计算高质量建筑的邻接程度）
        high_quality_clustering = self._calculate_clustering(G_after, threshold=2)

        # 4. 财政效率
        num_subsidized = len(self.policy_log['subsidies_granted'])
        total_expenditure = self.policy_log['total_expenditure']
        expenditure_per_improvement = total_expenditure / len(improved_buildings) if improved_buildings else 0

        metrics = {
            'num_improved': len(improved_buildings),
            'num_deteriorated': len(deteriorated_buildings),
            'improvement_rate': len(improved_buildings) / len(G_before.nodes()),
            'deterioration_rate': len(deteriorated_buildings) / len(G_before.nodes()),

            'avg_quality_before': np.mean(qualities_before),
            'avg_quality_after': np.mean(qualities_after),
            'quality_improvement': np.mean(qualities_after) - np.mean(qualities_before),

            'high_quality_clustering': high_quality_clustering,

            'num_subsidized_buildings': num_subsidized,
            'total_expenditure': total_expenditure,
            'remaining_budget': self.remaining_budget,
            'expenditure_per_improvement': expenditure_per_improvement,

            'quality_distribution_before': np.bincount(qualities_before, minlength=6)[1:],
            'quality_distribution_after': np.bincount(qualities_after, minlength=6)[1:]
        }

        return metrics

    def _calculate_clustering(self, G, threshold: int = 2) -> float:
        """
        计算高质量建筑的空间聚集度

        逻辑：高质量建筑（quality <= threshold）的邻居中，
             有多少比例也是高质量建筑

        Returns:
            clustering: 聚集度 [0, 1]
        """
        high_quality_nodes = [
            n for n in G.nodes()
            if G.nodes[n].get('quality', 5) <= threshold
        ]

        if not high_quality_nodes:
            return 0.0

        clustering_scores = []

        for node in high_quality_nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            high_quality_neighbors = [
                n for n in neighbors
                if G.nodes[n].get('quality', 5) <= threshold
            ]

            clustering_scores.append(len(high_quality_neighbors) / len(neighbors))

        return np.mean(clustering_scores) if clustering_scores else 0.0

    def print_impact_report(self, metrics: Dict):
        """打印政策影响报告"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║             政策影响评估报告                                  ║
║             场景: {self.policy_scenario:^20s}                      ║
╚══════════════════════════════════════════════════════════════╝

【质量改善】
  改善建筑数: {metrics['num_improved']} 栋
  恶化建筑数: {metrics['num_deteriorated']} 栋
  改善率: {metrics['improvement_rate']*100:.2f}%
  平均质量变化: {metrics['quality_improvement']:+.3f}

【质量分布变化】
  优质(1级): {metrics['quality_distribution_before'][0]} → {metrics['quality_distribution_after'][0]}
  良好(2级): {metrics['quality_distribution_before'][1]} → {metrics['quality_distribution_after'][1]}
  一般(3级): {metrics['quality_distribution_before'][2]} → {metrics['quality_distribution_after'][2]}
  较差(4级): {metrics['quality_distribution_before'][3]} → {metrics['quality_distribution_after'][3]}
  危房(5级): {metrics['quality_distribution_before'][4]} → {metrics['quality_distribution_after'][4]}

【空间效应】
  高质量建筑聚集度: {metrics['high_quality_clustering']:.3f}

【财政投入】
  补贴建筑数: {metrics['num_subsidized_buildings']} 栋
  总支出: {metrics['total_expenditure']/1e4:.2f} 万元
  剩余预算: {metrics['remaining_budget']/1e8:.2f} 亿元
  单建筑改善成本: {metrics['expenditure_per_improvement']/1e4:.2f} 万元

══════════════════════════════════════════════════════════════
        """)


# ==============================================================================
# 使用示例
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║               政府智能体 使用示例                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 示例1: 创建不同场景的政府Agent
    print("\n【示例1】创建三种政策场景的政府Agent")
    print("=" * 60)

    gov_trend = GovernmentAgent('trend', budget=1e8)
    gov_cbd = GovernmentAgent('cbd', budget=1e8, city_center_coords=(101.7782, 36.6171))
    gov_tod = GovernmentAgent('tod', budget=1e8)

    # 示例2: 生成政策描述
    print("\n【示例2】生成针对性的政策描述")
    print("=" * 60)

    # 模拟一个靠近CBD的低质量建筑
    building_near_cbd = {
        'building_id': 123,
        'quality': 4,  # 较差
        'distance_to_cbd': 1500,  # 1.5km
        'distance_to_metro': 3000,
        'use_type': '商业'
    }

    policy_text_cbd = gov_cbd.generate_policy_context(building_near_cbd)
    print(policy_text_cbd)

    # 示例3: 应用补贴
    print("\n【示例3】应用财政补贴")
    print("=" * 60)

    owner_decision = {
        'decision': '重大翻新',
        'expected_quality_change': 2,
        'estimated_cost': 45  # 45万元
    }

    subsidy = gov_cbd.apply_subsidy(123, owner_decision)
    print(f"业主决策: {owner_decision['decision']}, 成本: {owner_decision['estimated_cost']}万元")
    print(f"政府补贴: {subsidy}万元")
    print(f"业主实际支出: {owner_decision['estimated_cost'] - subsidy}万元")
    print(f"政府剩余预算: {gov_cbd.remaining_budget/1e8:.2f}亿元")

    # 示例4: 政策影响评估（需要真实图数据）
    print("\n【示例4】政策影响评估")
    print("=" * 60)
    print("（需要真实图数据，此处展示报告格式）")

    mock_metrics = {
        'num_improved': 1250,
        'num_deteriorated': 450,
        'improvement_rate': 0.52,
        'deterioration_rate': 0.19,
        'avg_quality_before': 3.2,
        'avg_quality_after': 2.8,
        'quality_improvement': -0.4,
        'high_quality_clustering': 0.67,
        'num_subsidized_buildings': 850,
        'total_expenditure': 2.1e7,
        'remaining_budget': 7.9e7,
        'expenditure_per_improvement': 1.68e4,
        'quality_distribution_before': np.array([120, 380, 920, 650, 230]),
        'quality_distribution_after': np.array([280, 520, 850, 480, 170])
    }

    gov_cbd.print_impact_report(mock_metrics)
