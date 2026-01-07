# train_router.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score
import numpy as np
import os

# --- 설정 ---
CSV_PATH = './cropped_images_metadata.csv'
IMAGE_DIR = './cropped_images'
MODEL_NAME = "google/vit-base-patch16-224-in21k"
OUTPUT_DIR = "./router_model"
# ------------

# 데이터셋 클래스
class RouterDataset(Dataset):
    def __init__(self, csv_path, image_dir, processor):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = processor
        
        # --- *** 핵심 수정 부분 *** ---
        # 클래스 이름 정의 (영어 -> 한글)
        self.class_map = {'top': '상의', 'bottom': '하의', 'outer': '아우터', 'onepiece': '원피스'}
        
        # 'category' 컬럼 대신, 'image_name'에서 진짜 메인 카테고리를 추출하여 라벨로 사용
        # 예: 'look1_outer_0.jpg' -> 'outer' 추출 -> '아우터'로 변환
        self.df['main_category'] = self.df['image_name'].apply(
            lambda x: self.class_map.get(x.split('_')[-2], '기타')
        )
        # ---------------------------

        # 새로운 main_category를 기준으로 숫자로 변환
        self.labels = self.df['main_category'].astype('category').cat.codes
        self.id2label = dict(enumerate(self.df['main_category'].astype('category').cat.categories))
        self.label2id = {v: k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_name'])
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return self.__getitem__((idx + 1) % len(self))
            
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs.pixel_values.squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# (이하 코드는 이전과 동일)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, preds)}

if __name__ == '__main__':
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    full_dataset = RouterDataset(CSV_PATH, IMAGE_DIR, processor)
    
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(full_dataset.id2label),
        id2label=full_dataset.id2label,
        label2id=full_dataset.label2id
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    print("1차 분류기 (Router Model) 재훈련을 시작합니다...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"🎉 1차 분류기 재훈련이 완료되었습니다! 모델이 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")