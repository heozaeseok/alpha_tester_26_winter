import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
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

print("--- Step 1: 격자 좌표 및 영역 분석 ---")
grid_df = pd.read_csv(grid_csv_path)
unique_lons = np.sort(grid_df['lon'].unique())
unique_lats = np.sort(grid_df['lat'].unique())
n_lon, n_lat = len(unique_lons), len(unique_lats)
res = 20.0

# 격자 중심점 기반의 경계(Extent) 계산
left, right = unique_lons[0] - res/2, unique_lons[-1] + res/2
bottom, top = unique_lats[0] - res/2, unique_lats[-1] + res/2

# ==========================================
# 2. Step 2: DEM 정밀 리샘플링 및 추출
# ==========================================
print("--- Step 2: 고도 데이터 정밀 매핑 중 ---")
with rasterio.open(dem_path) as src:
    # DEM의 좌표계가 정의되지 않은 경우 EPSG:5179 강제 적용
    src_crs = src.crs if src.crs else 'EPSG:5179'
    
    # 출력용 transform 생성 (Top-Left 기준)
    dst_transform = from_origin(left, top, res, res)
    elevation_2d = np.zeros((n_lat, n_lon), dtype=np.float32)

    # 데이터 리샘플링을 통한 좌표 정합성 확보
    reproject(
        source=rasterio.band(src, 1),
        destination=elevation_2d,
        src_transform=src.transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs='EPSG:5179',
        resampling=Resampling.bilinear
    )

# 결측치(NoData) 처리 (비정상적으로 낮은 값 제거)
elevation_2d[elevation_2d < -100] = np.nan
elevation_2d = pd.DataFrame(elevation_2d).ffill().bfill().values

# 경사 및 사면향 계산
dy, dx = np.gradient(elevation_2d, res)
slope_2d = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
aspect_2d = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360

# ==========================================
# 3. Step 3 & 4: 식생 데이터 매핑 (이미 검증됨)
# ==========================================
print("--- Step 3 & 4: 식생 및 수목 정보 통합 ---")
gdf = gpd.read_file(forest_shp_path, encoding='cp949')
target_col = 'STOR_TP_CD' if 'STOR_TP_CD' in gdf.columns else 'FRTP_CD'
gdf['FRTP_VAL'] = pd.to_numeric(gdf[target_col], errors='coerce').fillna(0).astype(int)

forest_2d = rasterize(
    ((geom, val) for geom, val in zip(gdf.geometry, gdf['FRTP_VAL'])),
    out_shape=(n_lat, n_lon),
    transform=dst_transform,
    fill=0
)

# [중요] CSV 데이터는 보통 Bottom-Up이므로 시각화와 저장 시 주의
# 데이터 통합용 (CSV 행 순서에 맞춤)
elev_flat = np.flipud(elevation_2d).flatten('F')
slope_flat = np.flipud(slope_2d).flatten('F')
aspect_flat = np.flipud(aspect_2d).flatten('F')
forest_flat = np.flipud(forest_2d).flatten('F')

final_df = grid_df.sort_values(by=['lon', 'lat']).copy()
final_df['elevation'] = elev_flat
final_df['slope'] = slope_flat
final_df['aspect'] = aspect_flat
final_df['forest_type'] = forest_flat

# 보정 로직
final_df.loc[(final_df['forest_type'] == 0) & (final_df['is_tree'] == 1), 'forest_type'] = 4
final_df.to_csv('subongsan_integrated_final.csv', index=False)

# ==========================================
# 4. Step 5: 시각화 검증
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
extent = [left, right, bottom, top]

# 고도 시각화
im1 = axes[0, 0].imshow(elevation_2d, cmap='terrain', extent=extent)
axes[0, 0].set_title('1. Elevation (m) - Reprojected')
plt.colorbar(im1, ax=axes[0, 0])

# 경사 시각화
im2 = axes[0, 1].imshow(slope_2d, cmap='magma', extent=extent)
axes[0, 1].set_title('2. Slope (Degree)')
plt.colorbar(im2, ax=axes[0, 1])

# 사면향 시각화
im3 = axes[1, 0].imshow(aspect_2d, cmap='hsv', extent=extent)
axes[1, 0].set_title('3. Aspect (Direction)')
plt.colorbar(im3, ax=axes[1, 0])

# 식생 시각화
colors = ['#ffffff', '#FF0000', '#008000', '#FFFF00', '#A9A9A9']
cmap_tree = ListedColormap(colors)
im4 = axes[1, 1].imshow(forest_2d, cmap=cmap_tree, extent=extent, vmin=0, vmax=4)
axes[1, 1].set_title('4. Forest Type (Corrected)')
cbar = plt.colorbar(im4, ax=axes[1, 1], ticks=[0.4, 1.2, 2.0, 2.8, 3.6])
cbar.ax.set_yticklabels(['None', 'Conifer', 'Deciduous', 'Mixed', 'Other'])

plt.tight_layout()
plt.savefig('subongsan_final_analysis_plot.png')
print("✅ 고도 정규화 완료. 이미지를 확인해 보세요.")
plt.show()