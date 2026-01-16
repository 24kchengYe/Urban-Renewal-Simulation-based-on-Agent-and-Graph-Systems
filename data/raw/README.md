# 原始数据目录

## 📁 应放置的文件

将西宁市建筑Shapefile数据放在这里：

```
data/raw/
├── buildings_2022.shp
├── buildings_2022.shx
├── buildings_2022.dbf
├── buildings_2022.prj
├── buildings_2023.shp
├── buildings_2023.shx
├── buildings_2023.dbf
├── buildings_2023.prj
├── buildings_2024.shp
├── buildings_2025.shp
└── ...
```

## 📋 必需字段

Shapefile必须包含以下字段：

| 字段名 | 类型 | 描述 | 示例值 |
|--------|------|------|--------|
| quality | Integer | 建筑质量等级 | 1-5 (1=优质, 5=危房) |
| year_built | Integer | 建成年份 | 2005 |
| area | Float | 建筑面积（㎡） | 150.5 |
| floors | Integer | 楼层数 | 6 |
| use_type | String | 用途 | "住宅", "商业", "混合" |
| last_renovation | Integer | 上次翻新年份 | 2015 |

## 🔧 可选字段

如果有以下字段更好（否则会自动生成默认值）：

- `building_id`: 建筑唯一标识
- `address`: 地址
- `owner_type`: 业主类型

## ⚠️ 注意事项

1. **坐标系**: 建议使用WGS84 (EPSG:4326)
2. **编码**: 确保中文字段使用UTF-8编码
3. **几何类型**: Polygon或MultiPolygon
4. **数据完整性**: 避免NULL值，用默认值填充

## 📖 数据处理示例

如果你的数据格式不符合要求，可以使用以下代码转换：

```python
import geopandas as gpd

# 读取原始数据
gdf = gpd.read_file("your_original_data.shp")

# 重命名字段（如果需要）
gdf = gdf.rename(columns={
    '质量等级': 'quality',
    '建筑年代': 'year_built',
    '建筑面积': 'area',
    '层数': 'floors',
    '用途': 'use_type'
})

# 映射用途类型（如果是数字编码）
use_type_map = {1: '住宅', 2: '商业', 3: '混合'}
gdf['use_type'] = gdf['use_type'].map(use_type_map)

# 保存为标准格式
gdf.to_file("buildings_2022.shp", encoding='utf-8')
```

## 📦 示例数据

如果没有真实数据，运行`experiments/demo_end_to_end.py`会自动生成模拟数据用于演示。
