import pandas as pd
import os
import shutil
from tqdm import tqdm

# --- 설정 ---
CSV_PATH = './cropped_images_metadata.csv'
IMAGE_DIR = './cropped_images'
OUTPUT_DATA_DIR = './specialist_data'
ORIGINAL_DATA_PATH = './labeled_fashion_data(polygon_features).csv'
# ------------

print("데이터 준비를 시작합니다...")
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

cropped_df = pd.read_csv(CSV_PATH)
original_df = pd.read_csv(ORIGINAL_DATA_PATH)

cropped_df['original_image_identifier'] = cropped_df['original_image_identifier'].astype(str)
original_df['이미지 식별자'] = original_df['이미지 식별자'].astype(str)

# --- *** 핵심 수정 부분: '스타일_스타일' 컬럼도 가져오도록 추가 *** ---
label_cols = [col for col in original_df.columns if '_라벨_' in col]
# '스타일_스타일' 컬럼을 명시적으로 추가
required_cols = ['이미지 식별자', '스타일_스타일'] + label_cols
merged_df = pd.merge(cropped_df, original_df[required_cols], 
                     left_on='original_image_identifier', right_on='이미지 식별자', how='left')
# ---------------------------------------------------------------

CATEGORIES = ['상의', '하의', '아우터', '원피스']

for category in CATEGORIES:
    print(f"\n'{category}' 카테고리 데이터 처리 중...")
    
    filter_col = f'{category}_라벨_카테고리'
    if filter_col not in merged_df.columns:
        continue
    specialist_df = merged_df[pd.notna(merged_df[filter_col])].copy()

    if specialist_df.empty:
        continue

    # --- *** 핵심 수정 부분: '스타일_스타일' 컬럼도 관련 컬럼으로 포함 *** ---
    relevant_cols = ['image_name', '스타일_스타일'] + [col for col in specialist_df.columns if f'{category}_라벨_' in col]
    specialist_df = specialist_df[relevant_cols]
    # -------------------------------------------------------------
    
    specialist_df.columns = [col.replace(f'{category}_라벨_', '') for col in specialist_df.columns]
    specialist_df.columns = [col.replace('스타일_스타일', '스타일') for col in specialist_df.columns] # 이름 정리
    
    specialist_df.dropna(axis=1, how='all', inplace=True)

    if specialist_df.shape[1] <= 1:
        continue

    output_csv_path = os.path.join(OUTPUT_DATA_DIR, f'{category}_metadata.csv')
    specialist_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    output_image_dir = os.path.join(OUTPUT_DATA_DIR, category)
    os.makedirs(output_image_dir, exist_ok=True)
    
    for image_name in tqdm(specialist_df['image_name'], desc=f"'{category}' 이미지 복사 중"):
        source_path = os.path.join(IMAGE_DIR, image_name)
        dest_path = os.path.join(output_image_dir, image_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)

print("\n🎉 '스타일' 정보가 포함된 모든 전문 모델용 데이터 준비가 완료되었습니다!")