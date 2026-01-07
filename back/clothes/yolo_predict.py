# yolo 모델 실제 실행해보는 코드

from ultralytics import YOLO
from PIL import Image
import os


# 1. 훈련된 나만의 모델 경로
# 보통 'runs/detect/train/weights/best.pt' 형태
MODEL_PATH = './train5/weights/best.pt' 

# 2. 테스트하고 싶은 이미지 파일들의 경로
IMAGE_PATHS = [
    './IMG_3260.jpg', 
    './IMG_3261.jpg'
]

# 3. 결과 이미지를 저장할 폴더
OUTPUT_DIR = 'test_results'

model = YOLO(MODEL_PATH)
# -----------------------------------------

def predict_my_clothes():
    """
    훈련된 YOLO 모델을 불러와 새로운 이미지들을 예측하고,
    좌표 값과 시각화된 결과 이미지를 저장합니다.
    """
    # 결과 저장 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 훈련된 모델 불러오기
    try:
        model = YOLO(MODEL_PATH)
        print(f"'{MODEL_PATH}' 모델을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"오류: 모델을 불러오는 데 실패했습니다. 경로를 확인해주세요. 에러: {e}")
        return

    # 2. 이미지들을 하나씩 모델에 넣어 예측 실행
    for img_path in IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"경고: '{img_path}' 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        print(f"\n--- '{img_path}' 이미지 분석 시작 ---")
        
        # 모델 예측
        results = model(img_path)

        # 3. 결과에서 좌표 값과 정보 추출하기
        for r in results:
            boxes = r.boxes
            if len(boxes) == 0:
                print("-> 탐지된 객체가 없습니다.")
                continue

            for box in boxes:
                # 클래스 이름 (상의, 하의 등)
                class_name = model.names[int(box.cls[0])]
                
                # 좌표 값 (x1, y1, x2, y2)
                coordinates = [int(x) for x in box.xyxy[0].tolist()]
                
                # 신뢰도 점수 (모델이 얼마나 확신하는지)
                confidence = float(box.conf[0])
                
                # 터미널에 결과 출력
                print(f"-> 탐지된 객체: {class_name}")
                print(f"   좌표 (x1, y1, x2, y2): {coordinates}")
                print(f"   신뢰도: {confidence:.2f}") # 소수점 2자리까지
        
        # 4. 시각화된 결과 이미지 저장하기
        # results 객체의 plot() 메소드는 바운딩 박스가 그려진 이미지를 반환합니다.
        result_image_array = r.plot() 
        result_image = Image.fromarray(result_image_array[..., ::-1]) # BGR -> RGB 변환
        
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        result_image.save(output_path)
        print(f"-> 분석 결과 이미지를 '{output_path}'에 저장했습니다.")


# pipeline에서 사용할 함수.
def find_clothes(image_path):
    """이미지에서 옷의 위치 좌표 리스트를 반환합니다."""
    results = model(image_path)
    boxes = []
    for r in results:
        for box in r.boxes:
            coordinates = [int(x) for x in box.xyxy[0].tolist()]
            boxes.append(coordinates)
    return boxes


if __name__ == '__main__':
    predict_my_clothes()