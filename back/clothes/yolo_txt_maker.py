# yolo 학습에 필요한 .txt파일 생성 코드

import pandas as pd
import os

def create_yolo_dataframe(csv_path):
    """
    원본 CSV 파일에서 YOLO 학습에 필요한 컬럼만 추출하여
    새로운 데이터프레임(train_yolo_df)을 생성합니다.
    """
    print(f"'{csv_path}' 파일을 읽는 중입니다...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다.")
        return None

    # YOLO 학습에 필요한 컬럼만 담을 리스트 초기화
    yolo_data_list = []
    item_types = ['상의', '하의', '아우터', '원피스']

    print("YOLO 학습에 필요한 데이터 추출을 시작합니다...")
    # 원본 데이터프레임 순회
    for index, row in df.iterrows():
        image_identifier = str(row['이미지 식별자'])
        image_width = row['이미지 너비']
        image_height = row['이미지 높이']

        for item_type in item_types:
            x_col, y_col, w_col, h_col = f'{item_type}_렉트_X좌표', f'{item_type}_렉트_Y좌표', f'{item_type}_렉트_가로', f'{item_type}_렉트_세로'

            # 해당 아이템의 좌표 정보가 있는지 확인 (0보다 큰 경우)
            if x_col in row and row.get(x_col, 0) > 0:
                yolo_data_list.append({
                    'image_identifier': image_identifier,
                    'image_width': image_width,
                    'image_height': image_height,
                    'class': item_type,
                    'x': row[x_col],
                    'y': row[y_col],
                    'width': row[w_col],
                    'height': row[h_col]
                })

    # 새로운 데이터프레임 생성
    train_yolo_df = pd.DataFrame(yolo_data_list)
    print("추출 완료! 'train_yolo_df' 데이터프레임을 생성했습니다.")
    return train_yolo_df


def create_yolo_label_files(yolo_df, output_dir='labels'):
    """
    train_yolo_df를 입력받아 YOLO 형식의 .txt 라벨 파일을 생성합니다.
    """
    if yolo_df is None or yolo_df.empty:
        print("데이터프레임이 비어있어 라벨 파일을 생성할 수 없습니다.")
        return

    # 라벨을 저장할 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 클래스 이름을 숫자 ID로 매핑 (0부터 시작)
    class_to_id = {name: i for i, name in enumerate(yolo_df['class'].unique())}
    print("\n클래스 매핑 정보:")
    print(class_to_id)

    print(f"\n'{output_dir}' 폴더에 YOLO 라벨 파일 생성을 시작합니다...")
    
    # 이미지 식별자별로 그룹화하여 처리
    for image_id, group in yolo_df.groupby('image_identifier'):
        label_path = os.path.join(output_dir, f"{image_id}.txt")
        
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                # 이미지 크기
                img_w, img_h = row['image_width'], row['image_height']
                
                # Bounding Box 좌표 (x, y, 너비, 높이)
                x, y, w, h = row['x'], row['y'], row['width'], row['height']

                # YOLO 형식으로 변환 (중심 x, 중심 y, 너비, 높이) 및 정규화
                center_x = (x + w / 2) / img_w
                center_y = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # 클래스 ID 가져오기
                class_id = class_to_id[row['class']]

                # 파일에 쓰기
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    print(f"라벨 파일 생성이 완료되었습니다. 총 {len(yolo_df.groupby('image_identifier'))}개의 파일이 생성되었습니다.")
    # YOLO 학습 설정에 필요한 YAML 파일 내용 출력
    print("\n--- YOLO 학습용 YAML 파일에 다음 내용을 추가하세요 ---")
    print("names:")
    for name, class_id in class_to_id.items():
        print(f"  {class_id}: {name}")
    print(f"nc: {len(class_to_id)}")
    print("-------------------------------------------------")


# --- 메인 코드 실행 ---
if __name__ == '__main__':
    # 1. 원본 CSV 파일에서 필요한 데이터만 추출하여 DataFrame 생성
    csv_file = 'C:/Users/breath03/Desktop/Smart_Closet/labeled_fashion_data(polygon_features).csv'
    train_yolo_df = create_yolo_dataframe(csv_file)

    # 생성된 DataFrame 확인 (상위 5개)
    if train_yolo_df is not None:
        print("\n생성된 train_yolo_df 미리보기:")
        print(train_yolo_df.head())

        # 2. 생성된 DataFrame을 바탕으로 YOLO .txt 라벨 파일들 생성
        create_yolo_label_files(train_yolo_df, output_dir='yolo_labels')