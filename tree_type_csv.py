import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Point

# 1. 파일 경로 설정
csv_path = r"C:\Users\USER\Desktop\forest_fire\SubongSan_Grid_ver4.csv"
shp_path = r"C:\Users\USER\Desktop\forest_fire\376112 (1)\376112.shp"
img_path = r"C:\Users\USER\Desktop\forest_fire\376112 (1)\37611.img"

# 2. 데이터 로드
df = pd.read_csv(csv_path)
gdf_shp = gpd.read_file(shp_path)

# 3. 고도(Elevation), 경사(Slope), 경사향(Aspect) 추출
with rasterio.open(img_path) as dem:
    # 좌표 리스트 생성
    coords = [(x, y) for x, y in zip(df['lon'], df['lat'])]
    
    # 고도 샘플링
    df['elevation'] = [val[0] for val in dem.sample(coords)]
    
    # 경사/경사향 계산을 위한 래스터 처리
    elev_array = dem.read(1)
    res_x, res_y = dem.res
    
    # 넘파이 경사도 계산 (가장자리 무시)
    dy, dx = np.gradient(elev_array, res_y, res_x)
    slope_array = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
    aspect_array = (np.arctan2(-dx, dy) * (180 / np.pi) + 360) % 360
    
    # 각 포인트에 해당하는 래스터 인덱스 찾기
    rows, cols = rasterio.transform.rowcol(dem.transform, df['lon'], df['lat'])
    
    # 인덱스 범위 제한 및 값 할당
    rows = np.clip(rows, 0, dem.height - 1)
    cols = np.clip(cols, 0, dem.width - 1)
    df['slope'] = slope_array[rows, cols]
    df['aspect'] = aspect_array[rows, cols]

# 4. 식생 정보(Forest Type) 매핑
# 공간 결합을 위해 CSV를 GeoDataFrame으로 변환
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf_csv = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf_shp.crs)

# 임상도와 결합하여 FIFTH_FRTP 정보 가져오기
joined = gpd.sjoin(gdf_csv, gdf_shp[['FIFTH_FRTP', 'geometry']], how='left', predicate='intersects')

def map_forest_type(row):
    if row['is_tree'] == 0:
        return 0
    
    frtp = str(row['FIFTH_FRTP']).strip()
    conifers = ['C', 'D', 'PD', 'PK', 'PL', 'PR', 'Cr', 'Co', 'Ab', 'Pc', 'PT']
    deciduous = ['H', 'Q', 'PQ', 'Po', 'Ca', 'PH']
    mixed = ['M']
    
    if frtp in conifers: return 1
    elif frtp in deciduous: return 2
    elif frtp in mixed: return 3
    else: return 4

joined['forest_type'] = joined.apply(map_forest_type, axis=1)

# 5. 최종 결과 정리 및 저장
# 불필요한 컬럼 제거 및 이름 변경
final_df = joined.drop(columns=['geometry', 'index_right', 'FIFTH_FRTP', 'tree_type'], errors='ignore')
final_df = final_df.rename(columns={'forest_type': 'tree_type'}) # 기존 양식 유지 시

# 소수점 보존 및 저장
final_df.to_csv('SubongSan_Grid_Final.csv', index=False, encoding='utf-8-sig')
final_df.to_excel('SubongSan_Grid_Final.xlsx', index=False)

print("모든 정보가 통합된 최종 파일(SubongSan_Grid_Final.csv/xlsx)이 생성되었습니다.")