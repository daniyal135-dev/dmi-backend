"""
TEXT DETECTION MODEL — AIGeneratedTextDetector
Fine-tuned transformer encoder + linear head for Human vs AI-generated binary classification.

Backbone: Hugging Face AutoModel (default microsoft/deberta-v3-base).
Older checkpoints trained with roberta-base still load if checkpoint['model_name'] == 'roberta-base'.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


DEFAULT_TEXT_ENCODER = "microsoft/deberta-v3-base"


class AIGeneratedTextDetector(nn.Module):
    """
    Encoder (DeBERTa-v3-base by default) + dropout + 2-way classifier.
    Pooling: first token hidden state (same convention as prior RoBERTa setup).
    """

    def __init__(self, num_classes=2, model_name=DEFAULT_TEXT_ENCODER, dropout=0.3):
        super().__init__()
        self.model_name = model_name
        self.dropout_rate = dropout

        try:
            self.backbone = AutoModel.from_pretrained(model_name)
            print(f"✓ Loaded backbone: {model_name}")
        except Exception as e:
            print(f"✗ Error loading backbone: {str(e)}")
            print("  Falling back to random init from config (NOT RECOMMENDED)")
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        print(f"✓ Classification head: {hidden_size} → {num_classes}")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def get_attention_weights(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        return outputs.attentions

    def save_model(self, save_path):
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_classes": 2,
                "model_name": self.model_name,
                "dropout": self.dropout_rate,
            },
            save_path,
        )
        print(f"✓ Model saved to: {save_path}")

    @classmethod
    def load_model(cls, load_path, device="cpu"):
        checkpoint = torch.load(load_path, map_location=device)
        model = cls(
            num_classes=checkpoint.get("num_classes", 2),
            model_name=checkpoint.get("model_name", DEFAULT_TEXT_ENCODER),
            dropout=checkpoint.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"✓ Model loaded from: {load_path}")
        return model
