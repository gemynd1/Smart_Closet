import pandas as pd
import os
from tqdm import tqdm

# --- 설정 ---
CSV_PATH = './labeled_fashion_data.csv' # 폴리곤 정보가 있는 원본 CSV
OUTPUT_LABEL_DIR = 'yolo_seg_labels'
# ------------

df = pd.read_csv(CSV_PATH)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

class_names = ["상의", "하의", "아우터", "원피스"]
class_map = {name: i for i, name in enumerate(class_names)}

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    image_id = str(row['이미지 식별자'])
    img_w, img_h = row['이미지 너비'], row['이미지 높이']
    
    label_content = ""
    
    for part in class_names:
        poly_x_cols = sorted([col for col in df.columns if f'{part}_폴리곤_X좌표' in col])
        poly_y_cols = sorted([col for col in df.columns if f'{part}_폴리곤_Y좌표' in col])

        if not poly_x_cols or pd.isna(row[poly_x_cols[0]]):
            continue
            
        points = []
        for x_col, y_col in zip(poly_x_cols, poly_y_cols):
            if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                # 좌표 정규화 (0~1 사이의 값으로)
                norm_x = row[x_col] / img_w
                norm_y = row[y_col] / img_h
                points.append(f"{norm_x:.6f} {norm_y:.6f}")
        
        if points:
            class_id = class_map[part]
            label_content += f"{class_id} " + " ".join(points) + "\n"

    if label_content:
        with open(os.path.join(OUTPUT_LABEL_DIR, f"{image_id}.txt"), 'w') as f:
            f.write(label_content)

print(f"\n'YOLO Segmentation' 라벨 파일 생성이 '{OUTPUT_LABEL_DIR}' 폴더에 완료되었습니다.")