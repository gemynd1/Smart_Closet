# yolo 학습에 쓰일 이미지/라벨 폴더를 학습/검증용으로 나누는 코드

import os
import random
import shutil

# 원본 폴더 경로
image_dir = './yolo_images'
label_dir = './yolo_seg_labels'

# 새로 만들 폴더 경로
base_output_dir = './fashion_dataset'

# 검증 데이터셋 비율
val_split_ratio = 0.2
# -----------

def split_data():
    # 전체 이미지 파일 목록 가져오기
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(all_images)

    # 학습용과 검증용으로 나누기
    split_index = int(len(all_images) * (1 - val_split_ratio))
    train_files = all_images[:split_index]
    val_files = all_images[split_index:]

    print(f"총 이미지 수: {len(all_images)}")
    print(f"학습용 데이터 수: {len(train_files)}")
    print(f"검증용 데이터 수: {len(val_files)}")

    # 폴더 생성 함수
    def create_dirs(base, set_name):
        os.makedirs(os.path.join(base, 'images', set_name), exist_ok=True)
        os.makedirs(os.path.join(base, 'labels', set_name), exist_ok=True)

    # 파일 복사 함수
    def copy_files(files, set_name):
        for filename in files:
            basename = os.path.splitext(filename)[0]
            # 이미지 복사
            shutil.copy(os.path.join(image_dir, filename), 
                        os.path.join(base_output_dir, 'images', set_name, filename))
            # 라벨 복사
            shutil.copy(os.path.join(label_dir, f"{basename}.txt"), 
                        os.path.join(base_output_dir, 'labels', set_name, f"{basename}.txt"))

    # 폴더 생성 및 파일 복사 실행
    create_dirs(base_output_dir, 'train')
    create_dirs(base_output_dir, 'val')
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    print(f"\n'/{base_output_dir}/' 폴더에 학습/검증 데이터셋 생성을 완료했습니다.")

if __name__ == '__main__':
    split_data()