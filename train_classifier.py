import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    TrainingArguments,
    Trainer
)
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import os

# --- 설정 ---
CSV_PATH = 'cropped_images_metadata.csv'
IMAGE_DIR = 'cropped_images'
MODEL_NAME = "google/vit-base-patch16-224-in21k"
OUTPUT_DIR = "fashion_classifier_model"
# ------------

# 1. 데이터셋 클래스 정의
class FashionDataset(Dataset):
    def __init__(self, csv_path, image_dir, processor):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = processor
        self.category_labels = self.df['category'].astype('category').cat.codes
        self.style_labels = self.df['style'].astype('category').cat.codes
        self.id2cat = dict(enumerate(self.df['category'].astype('category').cat.categories))
        self.id2style = dict(enumerate(self.df['style'].astype('category').cat.categories))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_name'])
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError): # 파일이 없거나 깨졌을 경우
            print(f"경고: '{image_path}' 파일을 읽을 수 없어 건너뜁니다. 다음 이미지로 대체합니다.")
            return self.__getitem__((idx + 1) % len(self)) # 다음 이미지를 대신 반환

        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values.squeeze(),
            "category_labels": torch.tensor(self.category_labels[idx], dtype=torch.long),
            "style_labels": torch.tensor(self.style_labels[idx], dtype=torch.long)
        }

# 2. Multi-Task 모델 설계
class MultiTaskClassifier(nn.Module):
    def __init__(self, num_categories, num_styles):
        super().__init__()
        self.body = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.body.config.hidden_size
        self.category_head = nn.Linear(hidden_size, num_categories)
        self.style_head = nn.Linear(hidden_size, num_styles)
    def forward(self, pixel_values, category_labels=None, style_labels=None):
        features = self.body(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        category_logits = self.category_head(features)
        style_logits = self.style_head(features)
        if category_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_category = loss_fct(category_logits, category_labels)
            loss_style = loss_fct(style_logits, style_labels)
            total_loss = loss_category + loss_style
            return {"loss": total_loss, "category_logits": category_logits, "style_logits": style_logits}
        return {"category_logits": category_logits, "style_logits": style_logits}

# 3. Multi-Task Trainer 정의
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']

def compute_metrics(eval_pred):
    category_logits, style_logits = eval_pred.predictions
    category_labels, style_labels = eval_pred.label_ids
    cat_preds = np.argmax(category_logits, axis=1)
    style_preds = np.argmax(style_logits, axis=1)
    return {
        'category_accuracy': accuracy_score(category_labels, cat_preds),
        'style_accuracy': accuracy_score(style_labels, style_preds)
    }

# --- 메인 실행 ---
if __name__ == '__main__':
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    full_dataset = FashionDataset(CSV_PATH, IMAGE_DIR, processor)
    
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    num_categories = len(full_dataset.id2cat)
    num_styles = len(full_dataset.id2style)
    model = MultiTaskClassifier(num_categories, num_styles)


    # 훈련 규칙을 가장 단순하고 확실하게 변경
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=3, # 먼저 3 에포크로 짧게 훈련해서 성공하는지 확인
        logging_steps=50,
        # 체크포인트 관련 복잡한 기능은 모두 비활성화하여 안정성 확보
        save_strategy="no", 
        eval_strategy="epoch",
        load_best_model_at_end=False,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 훈련 시작!
    trainer.train()

    # --- '수동으로' 최종 모델을 확실하게 저장하는 부분 ---
    print("\n훈련이 완료되었습니다. 최종 모델을 직접 저장합니다...")
    
    # 훈련이 끝난 직후의 모델 상태를 지정된 폴더에 직접 저장합니다.
    trainer.save_model(OUTPUT_DIR)

    # 라벨 정보도 함께 저장합니다.
    torch.save({
        'id2cat': full_dataset.id2cat,
        'id2style': full_dataset.id2style
    }, os.path.join(OUTPUT_DIR, 'label_maps.pth'))

    print(f"최종 모델 저장이 완료되었습니다! 이제 '{OUTPUT_DIR}' 폴더 안의 'pytorch_model.bin' 파일을 사용하여 예측을 실행할 수 있습니다.")