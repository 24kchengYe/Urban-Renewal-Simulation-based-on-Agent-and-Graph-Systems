"""
==============================================================================
图构建工具 (Graph Builder)
==============================================================================
功能：将建筑Shapefile转换为NetworkX图或PyTorch Geometric数据对象

支持三种图构建策略：
1. Euclidean Distance Graph - 欧氏距离k近邻图
2. Street Network Graph - 街道网络可达性图
3. Visibility Graph - 视线可达性图（推荐，最符合破窗理论）
==============================================================================
"""

import os
import numpy as np
import geopandas as gpd
import networkx as nx
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    print("⚠️ osmnx未安装，街道网络图功能不可用")


class GraphBuilder:
    """建筑图构建器"""

    def __init__(self, graph_type='visibility', k_neighbors=8, max_distance=500):
        """
        Args:
            graph_type: 图类型 ['euclidean', 'street', 'visibility']
            k_neighbors: k近邻数量（euclidean模式）
            max_distance: 最大连接距离（米）
        """
        self.graph_type = graph_type
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance

    def build_graph(self, buildings_gdf, city_name=None):
        """
        构建建筑图

        Args:
            buildings_gdf: GeoDataFrame，包含建筑物矢量数据
            city_name: 城市名称（仅街道网络模式需要）

        Returns:
            G: NetworkX图对象
        """
        print(f"\n🏗️ 开始构建建筑图 (类型: {self.graph_type})...")
        print(f"   建筑物数量: {len(buildings_gdf)}")

        # 确保有质心列
        if 'centroid' not in buildings_gdf.columns:
            buildings_gdf['centroid'] = buildings_gdf.geometry.centroid

        # 根据类型选择构建方法
        if self.graph_type == 'euclidean':
            G = self._build_euclidean_graph(buildings_gdf)

        elif self.graph_type == 'street':
            if not HAS_OSMNX:
                raise ImportError("街道网络模式需要安装osmnx: pip install osmnx")
            if city_name is None:
                raise ValueError("街道网络模式需要提供city_name参数")
            G = self._build_street_network_graph(buildings_gdf, city_name)

        elif self.graph_type == 'visibility':
            G = self._build_visibility_graph(buildings_gdf)

        else:
            raise ValueError(f"未知图类型: {self.graph_type}")

        print(f"✅ 图构建完成!")
        print(f"   节点数: {G.number_of_nodes()}")
        print(f"   边数: {G.number_of_edges()}")
        print(f"   平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

        return G

    def _build_euclidean_graph(self, buildings_gdf):
        """
        方案A: 欧氏距离k近邻图

        逻辑：每个建筑与最近的k个建筑连边
        """
        print("   使用欧氏距离k近邻构建...")

        # 提取质心坐标
        centroids = buildings_gdf['centroid']
        coords = np.array([(p.x, p.y) for p in centroids])

        # 构建KD树加速查询
        tree = KDTree(coords)

        # 查找k近邻
        distances, indices = tree.query(coords, k=self.k_neighbors + 1)

        # 创建NetworkX图
        G = nx.Graph()

        # 添加节点（保留所有建筑属性）
        for idx, row in tqdm(buildings_gdf.iterrows(), total=len(buildings_gdf),
                              desc="     添加节点"):
            G.add_node(idx, **row.to_dict())

        # 添加边
        edge_count = 0
        for i in tqdm(range(len(coords)), desc="     添加边"):
            for j, dist in zip(indices[i][1:], distances[i][1:]):
                if dist <= self.max_distance:
                    # 边权重：距离衰减（高斯核）
                    weight = np.exp(-dist / 100)
                    G.add_edge(i, j, distance=dist, weight=weight)
                    edge_count += 1

        print(f"   添加了 {edge_count} 条边")
        return G

    def _build_street_network_graph(self, buildings_gdf, city_name):
        """
        方案B: 街道网络可达性图

        逻辑：只有通过街道网络可达（距离<阈值）的建筑才连边
        更符合破窗理论："沿街可见"的建筑才相互影响
        """
        print(f"   使用街道网络构建（城市: {city_name}）...")

        # 1. 下载城市街道网络
        print("     下载街道网络...")
        try:
            street_network = ox.graph_from_place(city_name, network_type='walk')
        except Exception as e:
            print(f"     ⚠️ 无法下载{city_name}的街道网络，尝试使用边界框...")
            # 使用建筑物范围的边界框
            bbox = buildings_gdf.total_bounds  # (minx, miny, maxx, maxy)
            street_network = ox.graph_from_bbox(
                bbox[3], bbox[1], bbox[2], bbox[0],  # north, south, east, west
                network_type='walk'
            )

        # 2. 投影到测量坐标系
        print("     投影坐标系...")
        street_network_proj = ox.project_graph(street_network)
        buildings_proj = buildings_gdf.to_crs(street_network_proj.graph['crs'])

        # 3. 将建筑物质心捕捉到最近的街道节点
        print("     捕捉建筑到街道节点...")
        buildings_proj['nearest_node'] = ox.distance.nearest_nodes(
            street_network_proj,
            buildings_proj['centroid'].x,
            buildings_proj['centroid'].y
        )

        # 4. 计算建筑物间的街道网络距离
        G = nx.Graph()

        # 添加节点
        for idx, row in tqdm(buildings_proj.iterrows(), total=len(buildings_proj),
                              desc="     添加节点"):
            G.add_node(idx, **row.to_dict())

        # 添加边（只连接街道距离<阈值的建筑）
        edge_count = 0
        print("     计算街道网络距离...")
        for i in tqdm(range(len(buildings_proj)), desc="     添加边"):
            node_i = buildings_proj.iloc[i]['nearest_node']

            for j in range(i + 1, len(buildings_proj)):
                node_j = buildings_proj.iloc[j]['nearest_node']

                try:
                    # 计算街道网络最短路径
                    street_dist = nx.shortest_path_length(
                        street_network_proj,
                        node_i,
                        node_j,
                        weight='length'
                    )

                    if street_dist <= self.max_distance:
                        # 高斯衰减权重
                        weight = np.exp(-street_dist / 200)
                        G.add_edge(i, j, distance=street_dist, weight=weight)
                        edge_count += 1

                except nx.NetworkXNoPath:
                    # 两点不可达，不连边
                    continue

        print(f"   添加了 {edge_count} 条边")
        return G

    def _build_visibility_graph(self, buildings_gdf):
        """
        方案C: 视线可达性图（推荐）

        逻辑：只有视线不被其他建筑阻挡的建筑对才相互影响
        最符合破窗理论："能看到的才影响我"
        """
        print("   使用视线可达性构建...")

        # 创建建筑物障碍物集合
        print("     构建障碍物几何...")
        obstacles = unary_union(buildings_gdf.geometry)

        G = nx.Graph()
        centroids = buildings_gdf['centroid']

        # 添加节点
        for idx, row in tqdm(buildings_gdf.iterrows(), total=len(buildings_gdf),
                              desc="     添加节点"):
            G.add_node(idx, **row.to_dict())

        # 遍历所有建筑对，检测视线
        edge_count = 0
        total_pairs = len(buildings_gdf) * (len(buildings_gdf) - 1) // 2

        with tqdm(total=total_pairs, desc="     检测视线可达性") as pbar:
            for i in range(len(buildings_gdf)):
                c1 = centroids.iloc[i]

                for j in range(i + 1, len(buildings_gdf)):
                    c2 = centroids.iloc[j]

                    # 距离过滤
                    dist = c1.distance(c2)
                    if dist > self.max_distance:
                        pbar.update(1)
                        continue

                    # 构造视线
                    sight_line = LineString([c1, c2])

                    # 检测视线是否被建筑阻挡
                    if self._is_line_of_sight_clear(sight_line, obstacles, c1, c2):
                        # 视线通畅，添加边
                        weight = np.exp(-dist / 100)
                        G.add_edge(i, j, distance=dist, weight=weight, visible=True)
                        edge_count += 1

                    pbar.update(1)

        print(f"   添加了 {edge_count} 条边 (视线通畅)")
        return G

    @staticmethod
    def _is_line_of_sight_clear(sight_line, obstacles, start_point, end_point):
        """
        检测视线是否通畅

        Args:
            sight_line: 视线LineString
            obstacles: 障碍物集合
            start_point: 起点
            end_point: 终点

        Returns:
            bool: True表示视线通畅
        """
        # 计算视线与障碍物的交集
        intersections = sight_line.intersection(obstacles)

        # 如果没有交集，视线通畅
        if intersections.is_empty:
            return True

        # 如果交集只是起点和终点（建筑本身），视线通畅
        if hasattr(intersections, 'geom_type'):
            if intersections.geom_type == 'Point':
                # 单个交点
                if intersections.equals(start_point) or intersections.equals(end_point):
                    return True
            elif intersections.geom_type == 'MultiPoint':
                # 多个交点（最多2个：起点+终点）
                if len(intersections.geoms) <= 2:
                    # 检查是否都是起点终点
                    for pt in intersections.geoms:
                        if not (pt.equals(start_point) or pt.equals(end_point)):
                            return False
                    return True

        # 其他情况：视线被阻挡
        return False

    def add_external_features(self, G, poi_df=None, flow_df=None, metro_gdf=None):
        """
        为图节点添加外部特征

        Args:
            G: NetworkX图
            poi_df: POI数据（包含building_id, poi_density_500m列）
            flow_df: 人流量数据（包含building_id, pedestrian_flow列）
            metro_gdf: 地铁站点数据（GeoDataFrame）

        Returns:
            G: 添加了特征的图
        """
        print("\n📊 添加外部特征...")

        # 1. 添加POI密度
        if poi_df is not None:
            print("   添加POI密度...")
            for idx in G.nodes():
                if idx in poi_df.index:
                    G.nodes[idx]['poi_density_500m'] = poi_df.loc[idx, 'poi_density_500m']
                else:
                    G.nodes[idx]['poi_density_500m'] = 0.0

        # 2. 添加人流量
        if flow_df is not None:
            print("   添加人流量数据...")
            for idx in G.nodes():
                if idx in flow_df.index:
                    G.nodes[idx]['pedestrian_flow'] = flow_df.loc[idx, 'pedestrian_flow']
                else:
                    G.nodes[idx]['pedestrian_flow'] = 0.0

        # 3. 计算到地铁站距离
        if metro_gdf is not None:
            print("   计算到地铁站距离...")
            for idx in G.nodes():
                building_point = G.nodes[idx]['centroid']
                # 计算到最近地铁站的距离
                distances = metro_gdf.geometry.distance(building_point)
                G.nodes[idx]['distance_to_metro'] = distances.min()

        # 4. 计算建筑密度（500米缓冲区内的建筑数量）
        print("   计算建筑密度...")
        for idx in G.nodes():
            # 获取500米范围内的邻居数量作为密度代理
            neighbors_500m = [
                n for n in G.neighbors(idx)
                if G[idx][n]['distance'] <= 500
            ]
            G.nodes[idx]['building_density'] = len(neighbors_500m) / (np.pi * 0.5**2)  # 建筑/km²

        print("✅ 外部特征添加完成")
        return G

    def save_graph(self, G, filepath):
        """保存图到文件"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(G, f)
        print(f"✅ 图已保存: {filepath}")

    @staticmethod
    def load_graph(filepath):
        """从文件加载图"""
        import pickle
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        print(f"✅ 图已加载: {filepath}")
        return G


# ==============================================================================
# 使用示例
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         GPGLB 图构建工具 使用示例                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 示例1: 从Shapefile构建视线可达性图
    print("\n【示例1】构建视线可达性图")
    print("=" * 60)

    # 假设你有建筑物Shapefile
    # buildings_gdf = gpd.read_file("data/raw/buildings_2022.shp")

    # 创建图构建器
    builder = GraphBuilder(
        graph_type='visibility',
        max_distance=300  # 300米视线范围
    )

    # 构建图
    # G = builder.build_graph(buildings_gdf)

    # 保存图
    # builder.save_graph(G, "data/processed/graph_2022.pkl")

    print("\n提示：取消注释上述代码并提供真实Shapefile路径即可运行")

    # 示例2: 加载已保存的图
    print("\n【示例2】加载已保存的图")
    print("=" * 60)
    # G = GraphBuilder.load_graph("data/processed/graph_2022.pkl")
    # print(f"节点数: {G.number_of_nodes()}")
    # print(f"边数: {G.number_of_edges()}")
