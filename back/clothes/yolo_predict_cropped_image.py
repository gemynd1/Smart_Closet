import os
from ultralytics import YOLO
from PIL import Image
import argparse


# 1. 훈련된 YOLO 모델의 경로.
YOLO_MODEL_PATH = 'C:/Users/breath03/runs/detect/train5/weights/best.pt'

# 2. 잘라낸 이미지를 저장할 폴더 이름
OUTPUT_CROPPED_DIR = 'pipeline_cropped_images'
# ------------

def run_yolo_and_crop(image_path):
    """
    하나의 이미지를 입력받아 YOLO로 객체를 탐지하고,
    탐지된 객체들을 각각의 이미지 파일로 잘라 저장합니다.
    """
    # 결과 저장 폴더 생성
    os.makedirs(OUTPUT_CROPPED_DIR, exist_ok=True)

    # 1. 훈련된 YOLO 모델 불러오기
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"'{YOLO_MODEL_PATH}' YOLO 모델을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"오류: YOLO 모델을 불러오는 데 실패했습니다. 경로를 확인해주세요. 에러: {e}")
        return

    # 2. 원본 이미지 열기
    try:
        original_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"오류: '{image_path}' 이미지 파일을 찾을 수 없습니다.")
        return

    print(f"\n--- '{image_path}' 이미지 분석 시작 ---")
    
    # 3. YOLO 모델로 예측 실행
    results = model(original_image)

    # 4. 결과에서 탐지된 객체들을 하나씩 처리
    detected_count = 0
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            print("-> 탐지된 객체가 없습니다.")
            continue

        for i, box in enumerate(boxes):
            # 클래스 이름 (상의, 하의 등)
            class_name = model.names[int(box.cls[0])]
            
            # 좌표 값 (x1, y1, x2, y2)
            coordinates = [int(x) for x in box.xyxy[0].tolist()]
            
            # 5. 좌표를 이용해 원본 이미지 자르기
            cropped_image = original_image.crop(coordinates)
            
            # 6. 잘라낸 이미지 저장
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            new_filename = f"{base_filename}_cropped_{i}_{class_name}.jpg"
            output_path = os.path.join(OUTPUT_CROPPED_DIR, new_filename)
            
            cropped_image.save(output_path)
            print(f"-> '{class_name}'(을)를 탐지하여 '{output_path}' 경로에 저장했습니다.")
            detected_count += 1
            
    if detected_count > 0:
        print(f"\n총 {detected_count}개의 객체를 잘라내어 저장했습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO 모델로 옷을 탐지하고 잘라냅니다.")
    parser.add_argument("--image", required=True, type=str, help="분석할 이미지 파일 경로")
    args = parser.parse_args()
    
    run_yolo_and_crop(args.image)