import os
import random
import shutil
from tqdm import tqdm # 진행 상황 표시를 위한 라이브러리

# --- 설정 ---
# 원본 폴더 경로
image_dir = './yolo_images'
label_dir = './yolo_seg_labels' # Segmentation 라벨 폴더를 사용하도록 수정

# 새로 만들 폴더 경로
base_output_dir = './fashion_dataset'

# 검증 데이터셋 비율
val_split_ratio = 0.2
# -----------

def split_data_safely():
    # --- 1. 경로 검사 ---
    print("--- 필요한 폴더가 있는지 확인합니다 ---")
    if not os.path.isdir(image_dir):
        print(f"🚨 오류: 이미지 폴더 '{image_dir}'를 찾을 수 없습니다.")
        print("스크립트가 'yolo_images' 폴더와 같은 위치에 있는지 확인해주세요.")
        return
    if not os.path.isdir(label_dir):
        print(f"🚨 오류: 라벨 폴더 '{label_dir}'를 찾을 수 없습니다.")
        print("스크립트가 'yolo_seg_labels' 폴더와 같은 위치에 있는지 확인해주세요.")
        return
    print("✅ 모든 폴더를 성공적으로 찾았습니다.\n")

    # --- 2. 파일 목록 가져오기 ---
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    if not all_images:
        print("🚨 오류: 이미지 폴더가 비어있습니다. 'yolo_images' 폴더에 이미지 파일이 있는지 확인해주세요.")
        return
        
    random.shuffle(all_images)

    # 학습용과 검증용으로 나누기
    split_index = int(len(all_images) * (1 - val_split_ratio))
    train_files = all_images[:split_index]
    val_files = all_images[split_index:]

    print(f"총 이미지 수: {len(all_images)}")
    print(f"학습용 데이터 수: {len(train_files)}")
    print(f"검증용 데이터 수: {len(val_files)}\n")

    # 폴더 생성 함수
    def create_dirs(base, set_name):
        os.makedirs(os.path.join(base, 'images', set_name), exist_ok=True)
        os.makedirs(os.path.join(base, 'labels', set_name), exist_ok=True)

    # 파일 복사 함수
    def copy_files(files, set_name):
        print(f"--- '{set_name}' 세트 파일 복사를 시작합니다 ---")
        for filename in tqdm(files, desc=f"'{set_name}' 복사 중"):
            basename = os.path.splitext(filename)[0]
            # 이미지 복사
            shutil.copy(os.path.join(image_dir, filename), 
                        os.path.join(base_output_dir, 'images', set_name, filename))
            # 라벨 복사
            source_label_path = os.path.join(label_dir, f"{basename}.txt")
            if os.path.exists(source_label_path):
                shutil.copy(source_label_path, 
                            os.path.join(base_output_dir, 'labels', set_name, f"{basename}.txt"))

    # 폴더 생성 및 파일 복사 실행
    create_dirs(base_output_dir, 'train')
    create_dirs(base_output_dir, 'val')
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    print(f"\n🎉 '{base_output_dir}' 폴더에 학습/검증 데이터셋 생성을 완료했습니다.")

if __name__ == '__main__':
    # tqdm 라이브러리가 없으면 설치 안내
    try:
        from tqdm import tqdm
    except ImportError:
        print("진행 상황을 표시하기 위해 'tqdm' 라이브러리가 필요합니다.")
        print("터미널에 'pip install tqdm'을 입력하여 설치해주세요.")
        exit()
        
    split_data_safely()