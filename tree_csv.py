import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==========================================
# 1. 파일 경로 및 격자 설정
# ==========================================
grid_csv_path = r'C:\Users\USER\Desktop\forest_fire\grid_analysis.csv'
dem_path = r'C:\Users\USER\Desktop\forest_fire\37611017\37611.img'
forest_shp_path = r'C:\Users\USER\Desktop\forest_fire\37611017\37611017.shp'

print("--- Step 1: 격자 좌표 분석 ---")
grid_df = pd.read_csv(grid_csv_path)
unique_lons = np.sort(grid_df['lon'].unique())
unique_lats = np.sort(grid_df['lat'].unique())
n_lon, n_lat = len(unique_lons), len(unique_lats)
res = 20.0

# 격자 중심점 기반의 경계 계산 (Top-Left 기준 transform 생성용)
left, right = unique_lons[0] - res/2, unique_lons[-1] + res/2
bottom, top = unique_lats[0] - res/2, unique_lats[-1] + res/2
dst_transform = from_origin(left, top, res, res)

# ==========================================
# 2. Step 2: DEM 데이터 래스터 매핑
# ==========================================
print("--- Step 2: 고도/경사/사면 데이터 생성 중 ---")
with rasterio.open(dem_path) as src:
    src_crs = src.crs if src.crs else 'EPSG:5179'
    elevation_2d = np.zeros((n_lat, n_lon), dtype=np.float32)

    reproject(
        source=rasterio.band(src, 1),
        destination=elevation_2d,
        src_transform=src.transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs='EPSG:5179',
        resampling=Resampling.bilinear
    )

elevation_2d[elevation_2d < -100] = np.nan
elevation_2d = pd.DataFrame(elevation_2d).ffill().bfill().values

# 지형 지수 계산
dy, dx = np.gradient(elevation_2d, res)
slope_2d = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
aspect_2d = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360

# ==========================================
# 3. Step 3: 식생 데이터 래스터화
# ==========================================
print("--- Step 3: 식생 정보 래스터화 중 ---")
gdf = gpd.read_file(forest_shp_path, encoding='cp949')
target_col = 'STOR_TP_CD' if 'STOR_TP_CD' in gdf.columns else 'FRTP_CD'
gdf['FRTP_VAL'] = pd.to_numeric(gdf[target_col], errors='coerce').fillna(0).astype(int)

forest_2d = rasterize(
    ((geom, val) for geom, val in zip(gdf.geometry, gdf['FRTP_VAL'])),
    out_shape=(n_lat, n_lon),
    transform=dst_transform,
    fill=0
)

# ==========================================
# 4. Step 4: CSV 데이터 매핑 (좌표 기반 매칭)
# ==========================================
print("--- Step 4: CSV 셀 단위 정밀 매핑 및 저장 ---")

# 2D 배열을 (lat, lon, value) 형태의 DataFrame으로 변환하는 함수
def raster_to_df(array, name):
    # 행(lat)은 위에서 아래로(top->bottom), 좌표는 bottom에서 top으로 정렬됨을 고려
    df_list = []
    for r in range(n_lat):
        for c in range(n_lon):
            df_list.append({
                'lat': unique_lats[::-1][r], # 위도는 역순(위에서 아래)
                'lon': unique_lons[c],
                name: array[r, c]
            })
    return pd.DataFrame(df_list)

# 각 레이어를 DF로 변환
df_elev = raster_to_df(elevation_2d, 'elevation')
df_slope = raster_to_df(slope_2d, 'slope')
df_aspect = raster_to_df(aspect_2d, 'aspect')
df_forest = raster_to_df(forest_2d, 'forest_type')

# 기존 grid_df와 좌표 기준으로 병합
final_df = grid_df.merge(df_elev, on=['lat', 'lon'], how='left') \
                  .merge(df_slope, on=['lat', 'lon'], how='left') \
                  .merge(df_aspect, on=['lat', 'lon'], how='left') \
                  .merge(df_forest, on=['lat', 'lon'], how='left')

# 보정 로직
final_df.loc[(final_df['forest_type'] == 0) & (final_df['is_tree'] == 1), 'forest_type'] = 4
final_df.to_csv('subongsan_integrated_final.csv', index=False)

# ==========================================
# 5. 시각화 검증
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
extent = [left, right, bottom, top]

axes[0, 0].imshow(elevation_2d, cmap='terrain', extent=extent)
axes[0, 1].imshow(slope_2d, cmap='magma', extent=extent)
axes[1, 0].imshow(aspect_2d, cmap='hsv', extent=extent)

colors = ['#ffffff', '#FF0000', '#008000', '#FFFF00', '#A9A9A9']
cmap_tree = ListedColormap(colors)
axes[1, 1].imshow(forest_2d, cmap=cmap_tree, extent=extent, vmin=0, vmax=4)

plt.tight_layout()
plt.show()
print("✅ CSV 매핑 및 시각화 완료.")