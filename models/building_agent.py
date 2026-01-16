"""
==============================================================================
建筑物业主智能体 (Building Owner Agent)
==============================================================================
功能：模拟不同类型的建筑物业主对建筑维护/翻新的决策行为

业主类型：
1. Resident（自住业主）: 保守，注重居住舒适度
2. Investor（投资者）: 激进，追求租金回报和房产增值
3. Commercial（商业业主）: 平衡，关注人流量和商业活力

决策机制：
- 感知：当前建筑状态 + 邻域环境 + POI密度 + 人流量
- 记忆：历史维护记录 + 预算约束
- 决策：重大翻新 | 一般维护 | 基础维修 | 维持现状 | 任其恶化
==============================================================================
"""

import numpy as np
import json
from typing import Dict, List, Optional


class BuildingOwnerAgent:
    """建筑物业主智能体"""

    def __init__(self, building_id: int, building_data: Dict,
                 owner_type: str = "resident", random_seed: int = None):
        """
        初始化业主Agent

        Args:
            building_id: 建筑物唯一标识
            building_data: 建筑物属性字典
            owner_type: 业主类型 ['resident', 'investor', 'commercial']
            random_seed: 随机种子（用于采样预算等）
        """
        self.building_id = building_id
        self.state = building_data  # 建筑物当前状态
        self.owner_type = owner_type

        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed + building_id)

        # 初始化业主个性
        self.persona = self._init_persona(owner_type)

        # 初始化记忆系统
        self.memory = {
            'last_renovation_year': building_data.get('last_renovation', 2000),
            'renovation_history': [],
            'budget_constraint': self._sample_budget(),
            'satisfaction_history': []
        }

    def _init_persona(self, owner_type: str) -> Dict:
        """
        定义业主个性

        Returns:
            persona: 包含业主特征的字典
        """
        personas = {
            'resident': {
                'risk_preference': 'conservative',    # 风险偏好：保守
                'priority': 'comfort',                 # 优先级：舒适度
                'sensitivity_to_neighbors': 0.7,       # 对邻居影响的敏感度
                'sensitivity_to_cost': 0.9,            # 对成本的敏感度
                'sensitivity_to_quality': 0.8,         # 对质量的敏感度
                'description': '自住业主，注重居住舒适度，决策保守，优先考虑生活质量而非投资回报'
            },
            'investor': {
                'risk_preference': 'aggressive',
                'priority': 'profit',                  # 优先级：利润
                'sensitivity_to_neighbors': 0.9,       # 高度关注邻域房价
                'sensitivity_to_cost': 0.6,            # 愿意承担更高成本
                'sensitivity_to_quality': 0.7,
                'description': '房产投资者，追求租金回报和房产增值，对市场变化敏感，愿意承担风险'
            },
            'commercial': {
                'risk_preference': 'balanced',
                'priority': 'revenue',                 # 优先级：营收
                'sensitivity_to_neighbors': 0.8,
                'sensitivity_to_cost': 0.7,
                'sensitivity_to_quality': 0.6,         # 功能性优先于美观
                'sensitivity_to_flow': 0.95,           # 极度关注人流量
                'description': '商业业主，关注人流量和商业活力，追求稳定营收，平衡投入产出'
            }
        }

        return personas.get(owner_type, personas['resident'])

    def _sample_budget(self) -> float:
        """
        根据业主类型采样年度维护预算

        Returns:
            budget: 预算（万元）
        """
        budget_ranges = {
            'resident': (5, 20),       # 自住：5-20万
            'investor': (10, 50),      # 投资：10-50万
            'commercial': (20, 100)    # 商业：20-100万
        }

        low, high = budget_ranges[self.owner_type]

        # 对数正态分布采样（更符合真实预算分布）
        mean_log = np.log((low + high) / 2)
        sigma_log = 0.5
        budget = np.random.lognormal(mean_log, sigma_log)

        # 约束到合理范围
        return np.clip(budget, low, high)

    def perceive_environment(self, neighbors_data: List[Dict],
                             external_features: Dict) -> Dict:
        """
        感知环境信息

        Args:
            neighbors_data: 邻居建筑数据列表
                [{'id': int, 'quality': int, 'distance': float}, ...]
            external_features: 外部环境特征
                {'poi_density': float, 'pedestrian_flow': float, ...}

        Returns:
            perception: 环境感知结果
        """
        # 计算邻域加权平均质量
        if neighbors_data:
            weighted_qualities = []
            total_weight = 0

            for neighbor in neighbors_data:
                dist = neighbor['distance']
                quality = neighbor['quality']

                # 高斯衰减权重
                weight = np.exp(-dist / 100)
                weighted_qualities.append(quality * weight)
                total_weight += weight

            neighbor_avg_quality = sum(weighted_qualities) / total_weight if total_weight > 0 else 3.0
        else:
            neighbor_avg_quality = 3.0  # 默认值

        # 计算邻域质量差距（破窗效应的核心）
        current_quality = self.state.get('quality', 3)
        quality_gap = neighbor_avg_quality - current_quality

        perception = {
            'neighbor_avg_quality': neighbor_avg_quality,
            'quality_gap': quality_gap,
            'num_neighbors': len(neighbors_data),
            **external_features  # 展开外部特征
        }

        return perception

    def generate_decision_context(self, perception: Dict,
                                  current_year: int,
                                  policy_context: str = "") -> str:
        """
        生成LLM决策所需的完整上下文（Prompt）

        Args:
            perception: 环境感知结果
            current_year: 当前年份
            policy_context: 政府政策描述

        Returns:
            prompt: 完整的决策Prompt
        """
        # 构建建筑状态描述
        current_quality = self.state.get('quality', 3)
        year_built = self.state.get('year_built', 2000)
        building_age = current_year - year_built

        quality_labels = {
            1: "优质（几乎全新）",
            2: "良好（维护较好）",
            3: "一般（有老化迹象）",
            4: "较差（需要维修）",
            5: "危房（严重老化）"
        }

        # 维护历史
        years_since_renovation = current_year - self.memory['last_renovation_year']

        prompt = f"""
你是一位{self.persona['description']}。

【建筑物当前状态】
- 建筑ID: {self.building_id}
- 当前质量等级: {current_quality}/5 - {quality_labels.get(current_quality, "未知")}
- 建成年份: {year_built}年（已使用{building_age}年）
- 建筑面积: {self.state.get('area', 0):.0f}㎡
- 楼层数: {self.state.get('floors', 0)}层
- 用途: {self.state.get('use_type', '未知')}
- 上次翻新: {self.memory['last_renovation_year']}年（距今{years_since_renovation}年）

【周边环境】
- 邻居建筑数量: {perception['num_neighbors']}栋
- 邻域平均质量: {perception['neighbor_avg_quality']:.2f}/5
- 质量差距: {perception['quality_gap']:+.2f}（正值=邻居更好，负值=我更好）
- POI密度(500m): {perception.get('poi_density_500m', 0):.1f}个/km²
- 日均客流量: {perception.get('pedestrian_flow', 0):.0f}人次
- 建筑密度: {perception.get('building_density', 0):.2f}栋/km²
- 距CBD: {perception.get('distance_to_cbd', 0):.0f}米
- 距地铁站: {perception.get('distance_to_metro', 9999):.0f}米

【业主状况】
- 业主类型: {self.owner_type}
- 风险偏好: {self.persona['risk_preference']}
- 核心诉求: {self.persona['priority']}
- 年度预算: {self.memory['budget_constraint']:.1f}万元

【政策环境】
{policy_context if policy_context else "无特殊的财政补贴或容积率激励政策"}

【决策任务】
根据以上信息，你需要决定未来1年内对建筑的维护/翻新策略。

决策选项及成本：
1. "重大翻新" - 质量+2级，成本30-50万元（全面改造，大幅提升）
2. "一般维护" - 质量+1级，成本10-30万元（常规维护，稳步提升）
3. "基础维修" - 质量持平，成本3-10万元（修补，防止恶化）
4. "维持现状" - 质量-1级，成本0万元（自然老化）
5. "任其恶化" - 质量-2级，成本0万元（完全忽视）

决策逻辑参考：
- **破窗效应**: 如果邻域质量远高于你的建筑（质量差距>1），可能产生翻新动力（不想拖后腿）
- **商业价值**: 如果客流量高但建筑质量差，商业价值未释放，值得投资（商业业主尤其关注）
- **预算约束**: 严格检查经济承受能力，重大翻新需>30万，一般维护需10-30万
- **政策激励**: 如有财政补贴/容积率奖励，可降低成本门槛或提高投资意愿
- **记忆影响**: 如果最近刚翻新过（<5年），可能倾向维持现状

请以JSON格式返回决策：
{{
    "decision": "重大翻新" | "一般维护" | "基础维修" | "维持现状" | "任其恶化",
    "expected_quality_change": -2 到 +2 的整数,
    "estimated_cost": 预计成本（万元）,
    "reasoning": "你的决策理由（100字内，需要提及：邻域影响、经济能力、政策因素、个人偏好）"
}}
"""

        return prompt

    def parse_llm_response(self, llm_output: str) -> Dict:
        """
        解析LLM返回的决策结果

        Args:
            llm_output: LLM返回的JSON字符串

        Returns:
            decision: 解析后的决策字典
        """
        try:
            # 尝试直接解析JSON
            decision = json.loads(llm_output)

            # 验证必要字段
            required_fields = ['decision', 'expected_quality_change', 'reasoning']
            for field in required_fields:
                if field not in decision:
                    raise ValueError(f"缺少必要字段: {field}")

            # 约束质量变化范围
            decision['expected_quality_change'] = np.clip(
                int(decision['expected_quality_change']), -2, 2
            )

            # 默认成本
            if 'estimated_cost' not in decision:
                cost_map = {
                    '重大翻新': 40,
                    '一般维护': 20,
                    '基础维修': 6,
                    '维持现状': 0,
                    '任其恶化': 0
                }
                decision['estimated_cost'] = cost_map.get(decision['decision'], 0)

            return decision

        except json.JSONDecodeError as e:
            print(f"⚠️ 建筑{self.building_id} LLM输出JSON解析失败: {e}")
            # 返回默认决策（维持现状）
            return {
                'decision': '维持现状',
                'expected_quality_change': -1,
                'estimated_cost': 0,
                'reasoning': 'LLM输出格式错误，默认维持现状'
            }

        except Exception as e:
            print(f"⚠️ 建筑{self.building_id} 决策解析异常: {e}")
            return {
                'decision': '维持现状',
                'expected_quality_change': -1,
                'estimated_cost': 0,
                'reasoning': f'异常: {str(e)}'
            }

    def update_state(self, decision: Dict, current_year: int):
        """
        根据决策更新建筑状态和Agent记忆

        Args:
            decision: LLM返回的决策
            current_year: 当前年份
        """
        # 应用质量变化
        old_quality = self.state.get('quality', 3)
        new_quality = np.clip(
            old_quality + decision['expected_quality_change'],
            1, 5
        )
        self.state['quality'] = int(new_quality)

        # 更新记忆
        if decision['decision'] in ['重大翻新', '一般维护']:
            self.memory['last_renovation_year'] = current_year

        self.memory['renovation_history'].append({
            'year': current_year,
            'decision': decision['decision'],
            'quality_change': decision['expected_quality_change'],
            'cost': decision['estimated_cost'],
            'reasoning': decision['reasoning']
        })

        # 计算满意度（简化版）
        satisfaction = self._calculate_satisfaction(decision, old_quality, new_quality)
        self.memory['satisfaction_history'].append({
            'year': current_year,
            'satisfaction': satisfaction
        })

    def _calculate_satisfaction(self, decision: Dict, old_quality: int,
                                new_quality: int) -> float:
        """
        计算业主对决策结果的满意度

        Returns:
            satisfaction: 满意度 [0, 1]
        """
        # 简化模型：质量提升越多越满意，成本越低越满意
        quality_improvement = new_quality - old_quality

        # 不同类型业主的满意度函数不同
        if self.owner_type == 'resident':
            # 自住业主：质量改善带来的满意度高
            satisfaction = 0.5 + quality_improvement * 0.2
        elif self.owner_type == 'investor':
            # 投资者：关注性价比
            cost = decision.get('estimated_cost', 0)
            roi = quality_improvement / (cost + 1)  # 避免除0
            satisfaction = 0.5 + roi * 0.3
        else:  # commercial
            # 商业业主：关注功能性
            satisfaction = 0.5 + quality_improvement * 0.15

        return np.clip(satisfaction, 0, 1)

    def get_state_summary(self) -> Dict:
        """
        获取Agent当前状态摘要

        Returns:
            summary: 状态摘要字典
        """
        return {
            'building_id': self.building_id,
            'owner_type': self.owner_type,
            'current_quality': self.state.get('quality', 3),
            'budget': self.memory['budget_constraint'],
            'last_renovation_year': self.memory['last_renovation_year'],
            'num_renovations': len(self.memory['renovation_history']),
            'avg_satisfaction': np.mean([
                h['satisfaction'] for h in self.memory['satisfaction_history']
            ]) if self.memory['satisfaction_history'] else 0.5
        }


# ==============================================================================
# 使用示例
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         建筑物业主智能体 使用示例                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 示例1: 创建不同类型的业主
    print("\n【示例1】创建三种类型的业主Agent")
    print("=" * 60)

    building_data = {
        'quality': 3,
        'year_built': 2005,
        'area': 150,
        'floors': 6,
        'use_type': '住宅',
        'last_renovation': 2010
    }

    # 自住业主
    resident_agent = BuildingOwnerAgent(1, building_data.copy(), 'resident')
    print(f"自住业主预算: {resident_agent.memory['budget_constraint']:.1f}万元")

    # 投资者
    investor_agent = BuildingOwnerAgent(2, building_data.copy(), 'investor')
    print(f"投资者预算: {investor_agent.memory['budget_constraint']:.1f}万元")

    # 商业业主
    commercial_agent = BuildingOwnerAgent(3, building_data.copy(), 'commercial')
    print(f"商业业主预算: {commercial_agent.memory['budget_constraint']:.1f}万元")

    # 示例2: 生成决策Prompt
    print("\n【示例2】生成决策Prompt")
    print("=" * 60)

    perception = {
        'neighbor_avg_quality': 2.5,
        'quality_gap': -0.5,  # 我比邻居好
        'num_neighbors': 8,
        'poi_density_500m': 15.3,
        'pedestrian_flow': 1200,
        'building_density': 45.2,
        'distance_to_cbd': 3500,
        'distance_to_metro': 450
    }

    prompt = resident_agent.generate_decision_context(
        perception,
        current_year=2023,
        policy_context="[新激励政策] 市政府为鼓励老旧小区改造，提供50%财政补贴"
    )

    print(prompt[:500] + "\n...\n")

    # 示例3: 解析LLM响应
    print("\n【示例3】解析LLM决策结果")
    print("=" * 60)

    mock_llm_response = """
    {
        "decision": "一般维护",
        "expected_quality_change": 1,
        "estimated_cost": 18,
        "reasoning": "虽然邻域质量略低，但政府提供50%补贴，成本可控。作为自住业主，希望保持舒适度，选择一般维护提升居住质量。"
    }
    """

    decision = resident_agent.parse_llm_response(mock_llm_response)
    print(f"决策: {decision['decision']}")
    print(f"质量变化: {decision['expected_quality_change']:+d}")
    print(f"成本: {decision['estimated_cost']}万元")
    print(f"理由: {decision['reasoning']}")

    # 示例4: 更新状态
    print("\n【示例4】更新Agent状态")
    print("=" * 60)

    print(f"更新前质量: {resident_agent.state['quality']}")
    resident_agent.update_state(decision, 2023)
    print(f"更新后质量: {resident_agent.state['quality']}")
    print(f"维护历史记录数: {len(resident_agent.memory['renovation_history'])}")
    print(f"状态摘要: {resident_agent.get_state_summary()}")
