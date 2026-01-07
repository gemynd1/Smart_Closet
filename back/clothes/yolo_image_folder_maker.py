# yolo모델에 학습에 쓰일 이미지 폴더 생성 코드

import os
import shutil
import pandas as pd

# 1. 원본 데이터 경로
original_image_root = 'C:/Users/user/Desktop/Smart_Closet_TEST/Training/원천데이터' # 원본 룩 사진들이 있는 최상위 폴더
label_folder = 'C:/Users/user/Desktop/Smart_Closet_TEST/yolo_seg_labels'             # 이전에 생성한 .txt 라벨 파일 폴더

# 2. 새로 생성될 YOLO 학습용 이미지 폴더
output_image_folder = 'C:/Users/user/Desktop/Smart_Closet_TEST/yolo_images'
# -----------------------------------------

def create_yolo_image_folder():
    """
    라벨(.txt) 파일이 존재하는 이미지들만 원본 폴더에서 찾아
    새로운 학습용 이미지 폴더로 복사합니다.
    """
    # 결과물 저장 폴더 생성
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # 1. 원본 이미지 경로 맵 생성
    print(f"'{original_image_root}'와 모든 하위 폴더에서 원본 이미지 파일을 찾는 중입니다...")
    if not os.path.isdir(original_image_root):
        print(f"오류: '{original_image_root}' 디렉토리를 찾을 수 없습니다.")
        return
        
    image_path_map = {}
    for root, _, files in os.walk(original_image_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                identifier = os.path.splitext(file)[0]
                image_path_map[identifier] = os.path.join(root, file)
    print(f"총 {len(image_path_map)}개의 원본 이미지 경로를 찾았습니다.")
    
    # 2. 라벨 파일 목록 가져오기
    if not os.path.isdir(label_folder):
        print(f"오류: '{label_folder}' 디렉토리를 찾을 수 없습니다. 이전 단계에서 라벨 파일을 먼저 생성해주세요.")
        return
        
    label_files = os.listdir(label_folder)
    print(f"'{label_folder}'에서 총 {len(label_files)}개의 라벨 파일을 찾았습니다.")

    # 3. 이미지 복사 시작
    copied_count = 0
    print(f"\n'{output_image_folder}' 폴더로 이미지 복사를 시작합니다...")
    for label_file in label_files:
        # 라벨 파일 이름에서 확장자를 제거하여 이미지 식별자 추출 (예: '994962')
        identifier = os.path.splitext(label_file)[0]

        # 해당 식별자를 가진 원본 이미지가 있는지 확인
        if identifier in image_path_map:
            source_path = image_path_map[identifier]
            # 복사될 파일의 전체 경로
            destination_path = os.path.join(output_image_folder, os.path.basename(source_path))
            
            # 이미지 파일 복사
            shutil.copyfile(source_path, destination_path)
            copied_count += 1
        
        if (copied_count % 1000 == 0) and (copied_count > 0):
             print(f"{copied_count}개의 이미지 복사 완료...")

    print("\n" + "="*50)
    print("🎉 YOLO 학습용 이미지 폴더 생성이 완료되었습니다! 🎉")
    print(f"🖼️  총 {copied_count}개의 이미지를 '{output_image_folder}' 폴더에 복사했습니다.")
    print("="*50)

# --- 메인 코드 실행 ---
if __name__ == '__main__':
    create_yolo_image_folder()