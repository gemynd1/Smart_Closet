# 폴리곤에 대한 값을 계산해서 새로운 피처로 추가하는 코드

import pandas as pd
from shapely.geometry import Polygon

# --- 피처 엔지니어링을 적용할 의류 카테고리 목록 ---
clothing_parts = ["아우터", "하의", "원피스", "상의"]

# 삭제할 원본 폴리곤 컬럼들을 저장할 리스트
all_poly_cols_to_drop = []

train_df = pd.read_csv('C:/Users/user/Desktop/Smart_Closet_TEST/labeled_fashion_data.csv')

# 각 의류 카테고리별로 반복 작업 수행
for part in clothing_parts:
    print(f"--- '{part}' 카테고리 폴리곤 피처 생성 시작 ---")

    # 1. 현재 카테고리에 해당하는 폴리곤 좌표 컬럼 이름들을 동적으로 찾기
    # 예: part가 '상의'이면 '상의_폴리곤_X좌표'가 포함된 컬럼을 찾음
    poly_x_cols = sorted([col for col in train_df.columns if f'{part}_폴리곤_X좌표' in col])
    poly_y_cols = sorted([col for col in train_df.columns if f'{part}_폴리곤_Y좌표' in col])
    
    # 해당 카테고리의 폴리곤 좌표 데이터가 없으면 다음 카테고리로 넘어감
    if not poly_x_cols:
        print(f"'{part}' 카테고리에 대한 폴리곤 좌표 정보가 없어 건너뜁니다.")
        continue

    # 삭제 리스트에 현재 좌표 컬럼들 추가
    all_poly_cols_to_drop.extend(poly_x_cols)
    all_poly_cols_to_drop.extend(poly_y_cols)
    
    # 각 행(이미지)마다 (X, Y) 좌표 쌍 리스트를 만드는 함수
    def create_polygon_points(row, x_cols, y_cols):
        points = []
        for x_col, y_col in zip(x_cols, y_cols):
            if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                points.append((row[x_col], row[y_col]))
        if len(points) >= 3:
            return Polygon(points)
        return None

    # 임시 'polygon' 컬럼 생성
    temp_polygon_col = f'{part}_polygon'
    train_df[temp_polygon_col] = train_df.apply(lambda row: create_polygon_points(row, poly_x_cols, poly_y_cols), axis=1)

    # 2. 새로운 피처(Feature)를 동적 컬럼명으로 계산 및 추가
    train_df[f'{part}_폴리곤_면적'] = train_df[temp_polygon_col].apply(lambda p: p.area if p else 0)

    def get_bbox_features(p):
        if p:
            minx, miny, maxx, maxy = p.bounds
            width = maxx - minx
            height = maxy - miny
            aspect_ratio = width / height if height > 0 else 0
            return width, height, aspect_ratio
        return 0, 0, 0

    bbox_features = train_df[temp_polygon_col].apply(get_bbox_features)
    train_df[[f'{part}_폴리곤_너비', f'{part}_폴리곤_높이', f'{part}_폴리곤_가로세로비']] = pd.DataFrame(bbox_features.tolist(), index=train_df.index)

    train_df[f'{part}_폴리곤_중심X'] = train_df[temp_polygon_col].apply(lambda p: p.centroid.x if p else 0)
    train_df[f'{part}_폴리곤_중심Y'] = train_df[temp_polygon_col].apply(lambda p: p.centroid.y if p else 0)
    
    # 임시 polygon 컬럼은 삭제 리스트에 추가
    all_poly_cols_to_drop.append(temp_polygon_col)
    print(f"'{part}' 카테고리 처리 완료.")


# 3. 모든 원본 폴리곤 좌표 컬럼 및 임시 컬럼들을 한 번에 제거
print("\n--- 불필요한 원본 폴리곤 좌표 컬럼 전체 삭제 중 ---")
train_df.drop(columns=all_poly_cols_to_drop, inplace=True, errors='ignore')

# 결과 확인
print("\n최종 DataFrame Info:")
train_df.info()

# 생성된 피처 중 일부를 확인
print("\n새롭게 생성된 피처 확인 (상위 5개):")
new_feature_cols = [col for col in train_df.columns if '_폴리곤_' in col]
print(train_df[['이미지 식별자'] + new_feature_cols].head())

# 파일로 저장
train_df.to_csv('labeled_fashion_data(polygon_features).csv', index=False, encoding='utf-8-sig')
print("\n'labeled_fashion_data(polygon_features).csv' 파일로 저장이 완료되었습니다.")