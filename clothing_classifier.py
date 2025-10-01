import torch
from PIL import Image
from transformers import AutoImageProcessor
from train_classifier import MultiTaskClassifier # 훈련 코드에서 모델 구조 가져오기
import os

MODEL_DIR = "fashion_classifier_model"
MODEL_NAME = "google/vit-base-patch16-224-in21k"

# 모델과 라벨 맵 미리 로드

# 원래 gpu있을때 사용하던 코드
# label_maps = torch.load(os.path.join(MODEL_DIR, 'label_maps.pth'))

# gpu 없어도 실행할 수 있도록 설정하는 부분.
label_maps = torch.load(os.path.join(MODEL_DIR, 'label_maps.pth'), map_location=torch.device('cpu'))

id2cat = label_maps['id2cat']
id2style = label_maps['id2style']

model = MultiTaskClassifier(num_categories=len(id2cat), num_styles=len(id2style))

# 원래 gpu있을때 사용하던 코드
# model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'pytorch_model.bin')))

# gpu 없어도 실행할 수 있도록 설정하는 부분.
device = torch.device('cpu')
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'pytorch_model.bin'), map_location=device))

model.eval()

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

def classify_clothing(cropped_image):
    """잘라낸 옷 이미지의 카테고리와 스타일을 반환합니다."""
    inputs = processor(images=cropped_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    pred_cat_id = outputs['category_logits'].argmax(-1).item()
    pred_style_id = outputs['style_logits'].argmax(-1).item()
    
    category = id2cat[pred_cat_id]
    style = id2style[pred_style_id]
    
    return category, style