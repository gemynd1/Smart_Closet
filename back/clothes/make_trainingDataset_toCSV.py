# 패션 라벨링 JSON 데이터를 CSV 파일로 만들어주는 코드

import os
import json
import pandas as pd
from tqdm import tqdm

# === 모든 의류 카테고리 데이터를 추출하도록 수정한 함수 ===
def flatten_json_data(data, category_name):
    """하나의 JSON 파일에서 아우터, 상의, 하의, 원피스 정보를 모두 추출하여 평탄화합니다."""
    flattened_data = {}

    # 1. 이미지 기본 정보 추출
    image_info = data.get('이미지 정보', {})
    flattened_data['이미지 식별자'] = image_info.get('이미지 식별자')
    flattened_data['이미지 높이'] = image_info.get('이미지 높이')
    flattened_data['이미지 파일명'] = image_info.get('이미지 파일명')
    flattened_data['이미지 너비'] = image_info.get('이미지 너비')

    # 2. 데이터셋 기본 정보 추출
    dataset_info = data.get('데이터셋 정보', {})
    flattened_data['파일 생성일자'] = dataset_info.get('파일 생성일자')
    flattened_data['카테고리'] = category_name # 파일이 속한 폴더명

    # 3. 데이터셋 상세설명 추출
    detail_info = dataset_info.get('데이터셋 상세설명', {})
    if not detail_info:
        return flattened_data

    # 3-1. 스타일 정보는 공통이므로 먼저 추출
    labeling_info = detail_info.get('라벨링', {})
    if labeling_info:
        style_list = labeling_info.get('스타일', [])
        if style_list and style_list[0]:
            for key, value in style_list[0].items():
                # 예: '스타일_스타일', '스타일_서브스타일'
                flattened_data[f'스타일_{key}'] = value

    # 3-2. 모든 의류 종류를 순회하며 데이터 추출
    clothing_parts = ["아우터", "하의", "원피스", "상의"]
    
    for part in clothing_parts:
        # 렉트 좌표 추출
        rect_list = detail_info.get('렉트좌표', {}).get(part, [])
        if rect_list and rect_list[0]:
            for key, value in rect_list[0].items():
                # 예: '상의_렉트_X좌표', '하의_렉트_Y좌표'
                flattened_data[f'{part}_렉트_{key}'] = value

        # 폴리곤 좌표 추출
        polygon_list = detail_info.get('폴리곤좌표', {}).get(part, [])
        if polygon_list and polygon_list[0]:
            for key, value in polygon_list[0].items():
                # 예: '아우터_폴리곤_X좌표1'
                flattened_data[f'{part}_폴리곤_{key}'] = value

        # 라벨링 정보 추출
        if labeling_info:
            label_list = labeling_info.get(part, [])
            if label_list and label_list[0]:
                for key, value in label_list[0].items():
                    # 예: '상의_라벨_카테고리', '하의_라벨_핏'
                    col_name = f'{part}_라벨_{key}'
                    if isinstance(value, list):
                        flattened_data[col_name] = ', '.join(map(str, value))
                    else:
                        flattened_data[col_name] = value

    return flattened_data


# 1. 기본 경로 설정
root_path = 'C:/Users/user/Desktop/Smart_Closet_TEST/Training/라벨링데이터'
all_flattened_data = []

# 2. os.walk로 하위 폴더와 파일 탐색
for dirpath, dirnames, filenames in tqdm(os.walk(root_path), desc="폴더 탐색 중"):
    if not dirnames:
        category_name = os.path.basename(dirpath)
        file_count = 0
        json_files = [f for f in filenames if f.endswith('.json')]
        
        print(f"\n'{category_name}' 카테고리 처리 시작... (총 {len(json_files)}개 파일 중 최대 5000개)")
        
        for filename in tqdm(json_files, desc=f"{category_name} 파일 처리 중"):
            if file_count >= 10000:
                break
            
            file_path = os.path.join(dirpath, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_content = json.load(f)
                
                flattened_record = flatten_json_data(json_content, category_name)
                all_flattened_data.append(flattened_record)
                
                file_count += 1
            
            except json.JSONDecodeError:
                print(f"경고: {file_path} 파일이 손상되었거나 JSON 형식이 아닙니다.")
            except Exception as e:
                print(f"에러 발생: {file_path} 처리 중 오류 - {e}")

# 5. 모든 데이터를 합쳐 최종 DataFrame 생성
print("\n모든 파일 처리 완료. DataFrame 생성 중...")
train_df = pd.DataFrame(all_flattened_data) # 변수명을 train_df로 변경

# 6. 결과 확인 및 저장
print("DataFrame 생성 완료!")
# info()를 통해 생성된 컬럼들을 확인해보세요. 상의_, 아우터_ 등의 컬럼이 보일 것입니다.
print(train_df.info()) 
print("\n--- 카테고리별 데이터 개수 ---")
print(train_df['카테고리'].value_counts())

train_df.to_csv('labeled_fashion_data.csv', index=False, encoding='utf-8-sig')
print("\n'labeled_fashion_data.csv' 파일로 저장이 완료되었습니다.")