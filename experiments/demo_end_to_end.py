"""
==============================================================================
GPGLB 端到端演示脚本
==============================================================================
这个脚本演示了完整的GPGLB工作流程：

1. 数据加载与图构建
2. LLM-GLN模拟（2022→2023校准）
3. 政策场景模拟（2025→2030）
4. 结果可视化和分析

注意：这是一个演示脚本，需要真实的建筑Shapefile数据才能运行
==============================================================================
"""

import sys
import os
import asyncio
import numpy as np
import geopandas as gpd
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_builder import GraphBuilder
from models.llm_gln import LLM_GLN_Simulator
from models.government_agent import GovernmentAgent


# ==============================================================================
# 阶段1: 数据准备
# ==============================================================================

def load_and_prepare_data():
    """
    加载建筑Shapefile和外部数据

    Returns:
        buildings_gdf: 建筑物GeoDataFrame
        poi_df: POI数据
        metro_gdf: 地铁站数据
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║              阶段1: 数据加载与预处理                          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 1. 加载建筑Shapefile
    print("\n📂 加载建筑Shapefile...")

    # TODO: 替换为你的真实数据路径
    buildings_shp_path = "./data/raw/buildings_2022.shp"

    if not os.path.exists(buildings_shp_path):
        print(f"❌ 文件不存在: {buildings_shp_path}")
        print("\n提示：请将西宁市建筑Shapefile放入 data/raw/ 目录")
        print("      文件应包含以下字段:")
        print("      - quality: 建筑质量等级 [1-5]")
        print("      - year_built: 建成年份")
        print("      - area: 建筑面积（㎡）")
        print("      - floors: 楼层数")
        print("      - use_type: 用途（住宅/商业/混合）")

        # 创建模拟数据用于演示
        print("\n🔧 创建模拟数据用于演示...")
        buildings_gdf = create_mock_buildings(n=500)
    else:
        buildings_gdf = gpd.read_file(buildings_shp_path)
        print(f"✅ 加载了 {len(buildings_gdf)} 个建筑")

    # 2. 加载POI数据（可选）
    print("\n📊 加载POI数据...")
    poi_csv_path = "./data/external/poi_xining.csv"

    if os.path.exists(poi_csv_path):
        poi_df = pd.read_csv(poi_csv_path)
        print(f"✅ 加载了 {len(poi_df)} 个POI")
    else:
        print("⚠️ POI数据不存在，将使用默认值")
        poi_df = None

    # 3. 加载地铁站数据（可选）
    print("\n🚇 加载地铁站数据...")
    metro_path = "./data/external/metro_stations.geojson"

    if os.path.exists(metro_path):
        metro_gdf = gpd.read_file(metro_path)
        print(f"✅ 加载了 {len(metro_gdf)} 个地铁站")
    else:
        print("⚠️ 地铁站数据不存在，将使用默认值")
        metro_gdf = None

    return buildings_gdf, poi_df, metro_gdf


def create_mock_buildings(n=500):
    """
    创建模拟建筑数据（仅用于演示）

    Args:
        n: 建筑数量

    Returns:
        buildings_gdf: 模拟的建筑GeoDataFrame
    """
    from shapely.geometry import Polygon, Point

    # 西宁市中心附近的随机点
    center_lon, center_lat = 101.7782, 36.6171

    data = []
    for i in range(n):
        # 随机位置（±0.05度，约5km范围）
        lon = center_lon + np.random.uniform(-0.05, 0.05)
        lat = center_lat + np.random.uniform(-0.05, 0.05)

        # 创建建筑多边形（简化为正方形）
        size = np.random.uniform(20, 100) / 111000  # 转换为度
        polygon = Polygon([
            (lon - size/2, lat - size/2),
            (lon + size/2, lat - size/2),
            (lon + size/2, lat + size/2),
            (lon - size/2, lat + size/2)
        ])

        # 建筑属性
        data.append({
            'geometry': polygon,
            'quality': np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3]),
            'year_built': np.random.randint(1990, 2020),
            'area': np.random.uniform(50, 300),
            'floors': np.random.randint(3, 15),
            'use_type': np.random.choice(['住宅', '商业', '混合'], p=[0.7, 0.2, 0.1]),
            'last_renovation': np.random.choice([2000, 2010, 2015, 2020])
        })

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    print(f"✅ 创建了 {n} 个模拟建筑")

    return gdf


# ==============================================================================
# 阶段2: 图构建
# ==============================================================================

def build_spatial_graph(buildings_gdf, poi_df=None, metro_gdf=None):
    """
    构建建筑空间图

    Returns:
        G: NetworkX图
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║              阶段2: 构建建筑空间图网络                        ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 创建图构建器
    builder = GraphBuilder(
        graph_type='visibility',  # 使用视线可达性图
        max_distance=300  # 300米视线范围
    )

    # 构建图
    G = builder.build_graph(buildings_gdf, city_name="西宁市")

    # 添加外部特征
    if poi_df is not None or metro_gdf is not None:
        G = builder.add_external_features(G, poi_df, None, metro_gdf)

    # 添加CBD距离（简化：到城市中心的距离）
    center_point = Point(101.7782, 36.6171)
    for nid in G.nodes():
        building_point = G.nodes[nid]['centroid']
        # 简化距离计算（度→米）
        dist_deg = building_point.distance(center_point)
        dist_m = dist_deg * 111000  # 粗略换算
        G.nodes[nid]['distance_to_cbd'] = dist_m

    # 保存图
    builder.save_graph(G, "./data/processed/graph_2022.pkl")

    return G


# ==============================================================================
# 阶段3: LLM-GLN模拟
# ==============================================================================

async def run_llm_simulation(G_2022, scenario='cbd'):
    """
    运行LLM-GLN模拟

    Args:
        G_2022: 2022年建筑图
        scenario: 政策场景

    Returns:
        graphs_history: 历史图字典
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║              阶段3: LLM-GLN模拟                               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 创建政府Agent
    print(f"\n🏛️ 创建政府Agent (场景: {scenario})...")
    gov_agent = GovernmentAgent(
        policy_scenario=scenario,
        budget=1e8,  # 1亿元预算
        city_center_coords=(101.7782, 36.6171)
    )

    # 创建LLM模拟器
    print("\n🤖 创建LLM-GLN模拟器...")
    simulator = LLM_GLN_Simulator(
        model="gpt-4o-mini",
        temperature=0.3,
        concurrency_limit=10  # 降低并发避免限流
    )

    # 运行模拟（2022→2023 校准）
    print("\n🎬 开始模拟...")
    graphs_history = await simulator.run_simulation(
        G_initial=G_2022,
        government_agent=gov_agent,
        start_year=2022,
        end_year=2023,  # 先模拟1年测试
        save_interval=1,
        save_dir="./results"
    )

    return graphs_history, gov_agent


# ==============================================================================
# 阶段4: 结果分析
# ==============================================================================

def analyze_results(G_before, G_after, gov_agent):
    """
    分析模拟结果
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║              阶段4: 结果分析与可视化                          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 评估政策影响
    metrics = gov_agent.evaluate_policy_impact(G_before, G_after)

    # 打印报告
    gov_agent.print_impact_report(metrics)

    # 可视化（简化版）
    print("\n📊 质量分布对比:")
    print("   质量等级 | 2022年 → 2023年")
    print("   " + "-" * 30)
    for i in range(1, 6):
        before_count = metrics['quality_distribution_before'][i-1]
        after_count = metrics['quality_distribution_after'][i-1]
        change = after_count - before_count
        print(f"   等级{i}    |  {before_count:4d}   →  {after_count:4d}  ({change:+4d})")

    return metrics


# ==============================================================================
# 主函数
# ==============================================================================

async def main():
    """主流程"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     GPGLB: Government and Resident Participation             ║
    ║            Graph Lingual Network for Buildings               ║
    ║                                                              ║
    ║            端到端演示脚本                                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 阶段1: 加载数据
    buildings_gdf, poi_df, metro_gdf = load_and_prepare_data()

    # 阶段2: 构建图
    G_2022 = build_spatial_graph(buildings_gdf, poi_df, metro_gdf)

    # 阶段3: LLM模拟
    graphs_history, gov_agent = await run_llm_simulation(G_2022, scenario='cbd')

    # 阶段4: 分析结果
    G_2023 = graphs_history[2023]
    metrics = analyze_results(G_2022, G_2023, gov_agent)

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              🎉 演示完成！                                    ║
    ╚══════════════════════════════════════════════════════════════╝

    下一步：
    1. 将真实的西宁市建筑Shapefile放入 data/raw/ 目录
    2. 修改脚本中的数据路径
    3. 运行完整的2022→2030模拟
    4. 对比三种政策场景（trend / cbd / tod）
    5. 生成可视化报告
    """)


if __name__ == "__main__":
    # 检查.env配置
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    if api_key == 'your_api_key_here' or not api_key:
        print("❌ 错误: 未配置OPENAI_API_KEY")
        print("   请在 .env 文件中设置你的API密钥")
        print("   复制 .env.example 为 .env 并编辑")
        exit(1)

    # 运行主流程
    asyncio.run(main())
