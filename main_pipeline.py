import argparse
from PIL import Image

# 1. 우리가 만든 전문가 모듈들을 불러옵니다.
from yolo_predict import find_clothes
from clothing_classifier import classify_clothing

def main_pipeline(image_path):
    """
    AI 서비스의 전체 파이프라인을 실행합니다.
    """
    print(f"--- 입력 이미지 분석 시작: {image_path} ---")
    
    # 2. YOLO 전문가에게 옷 위치를 찾아달라고 요청합니다.
    try:
        original_image = Image.open(image_path).convert("RGB")
        detected_boxes = find_clothes(image_path)
    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
        return
        
    if not detected_boxes:
        print("-> YOLO 모델이 이미지에서 옷을 찾지 못했습니다.")
        return

    print(f"-> 총 {len(detected_boxes)}개의 옷을 찾았습니다. 각 옷을 분석합니다...")

    # 3. 찾은 옷들을 하나씩 잘라내어 분류 전문가에게 넘겨줍니다.
    for i, box in enumerate(detected_boxes):
        # 원본 이미지에서 옷 부분만 잘라냅니다.
        cropped_image = original_image.crop(box)
        
        # 분류 전문가에게 카테고리와 스타일을 물어봅니다.
        category, style = classify_clothing(cropped_image)
        
        print(f"\n--- 분석 결과 #{i+1} ---")
        print(f"카테고리: {category}")
        print(f"스타일: {style}")
        print(f"위치: {box}")
        print("--------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="패션 이미지 분석 전체 파이프라인")
    parser.add_argument("--image", required=True, type=str, help="분석할 이미지 파일 경로")
    args = parser.parse_args()
    
    main_pipeline(args.image)