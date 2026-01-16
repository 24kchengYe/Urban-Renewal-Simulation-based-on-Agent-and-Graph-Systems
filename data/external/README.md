# 外部数据目录

## 📁 应放置的文件

将POI、人流量、地铁站等外部数据放在这里：

```
data/external/
├── poi_xining.csv           # POI数据
├── pedestrian_flow.csv      # 人流量数据
└── metro_stations.geojson   # 地铁站点数据
```

## 📋 数据格式

### 1. POI数据 (`poi_xining.csv`)

```csv
building_id,poi_density_500m,poi_category
0,15.3,"餐饮;购物;教育"
1,8.7,"餐饮;医疗"
2,23.1,"购物;娱乐;交通"
```

**字段说明：**
- `building_id`: 建筑ID（与Shapefile对应）
- `poi_density_500m`: 500米范围内POI密度（个/km²）
- `poi_category`: POI类别（可选）

### 2. 人流量数据 (`pedestrian_flow.csv`)

```csv
building_id,pedestrian_flow,peak_hour_flow
0,1200,350
1,450,120
2,2800,850
```

**字段说明：**
- `building_id`: 建筑ID
- `pedestrian_flow`: 日均人流量（人次）
- `peak_hour_flow`: 高峰小时人流量（可选）

### 3. 地铁站点数据 (`metro_stations.geojson`)

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [101.7782, 36.6171]
      },
      "properties": {
        "name": "西宁站",
        "line": "1号线",
        "open_year": 2023
      }
    }
  ]
}
```

## 🔧 数据采集方法

### POI数据

**方法1: 使用高德地图API**

```python
import requests

def get_poi_around_building(lon, lat, radius=500):
    """
    获取建筑物周边POI
    """
    url = "https://restapi.amap.com/v3/place/around"
    params = {
        'key': 'your_amap_key',
        'location': f'{lon},{lat}',
        'radius': radius,
        'output': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()

    return len(data['pois'])  # POI数量
```

**方法2: 使用OpenStreetMap**

```python
import osmnx as ox

def get_poi_osm(lat, lon, radius=500):
    """
    从OSM获取POI
    """
    tags = {'amenity': True, 'shop': True}
    gdf = ox.geometries_from_point((lat, lon), tags, dist=radius)
    return len(gdf)
```

### 人流量数据

**数据来源：**
- 移动运营商信令数据
- 百度慧眼/高德地图热力图
- 现场调研
- 摄像头客流统计

### 地铁站数据

**获取方式：**
- 城市公开数据平台
- OpenStreetMap
- 手动标注

```python
import geopandas as gpd
from shapely.geometry import Point

# 创建地铁站GeoDataFrame
metro_data = [
    {'name': '西宁站', 'line': '1号线', 'lon': 101.7782, 'lat': 36.6171},
    {'name': '新宁广场', 'line': '1号线', 'lon': 101.7850, 'lat': 36.6200},
]

geometry = [Point(d['lon'], d['lat']) for d in metro_data]
metro_gdf = gpd.GeoDataFrame(metro_data, geometry=geometry, crs='EPSG:4326')
metro_gdf.to_file('metro_stations.geojson', driver='GeoJSON')
```

## ⚠️ 注意事项

1. **数据对齐**: 确保building_id与Shapefile一致
2. **单位统一**: POI密度用个/km²，人流量用人次/天
3. **时间一致**: 外部数据的时间应与建筑Shapefile对应
4. **缺失值处理**: 用0或区域平均值填充

## 📖 数据处理示例

```python
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# 1. 计算建筑物周边的POI密度
buildings_gdf = gpd.read_file("../raw/buildings_2022.shp")
poi_gdf = gpd.read_file("poi_raw.geojson")

def calculate_poi_density(building_point, poi_gdf, radius=500):
    # 缓冲区
    buffer = building_point.buffer(radius / 111000)  # 度→米

    # 统计缓冲区内POI
    poi_count = poi_gdf[poi_gdf.intersects(buffer)].shape[0]

    # 密度（个/km²）
    area_km2 = (radius / 1000) ** 2 * 3.14159
    density = poi_count / area_km2

    return density

# 应用到所有建筑
buildings_gdf['poi_density_500m'] = buildings_gdf.geometry.centroid.apply(
    lambda p: calculate_poi_density(p, poi_gdf)
)

# 保存
buildings_gdf[['building_id', 'poi_density_500m']].to_csv('poi_xining.csv')
```

## 📦 如果没有外部数据

程序会自动使用默认值：
- `poi_density_500m`: 10.0
- `pedestrian_flow`: 500
- `distance_to_metro`: 9999（表示很远）

这不影响程序运行，但会降低模拟的真实性。
