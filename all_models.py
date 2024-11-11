from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Optional


class FastTextProcessor:
    """Text vectorization using TF-IDF"""

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts)


class EnsembleModel:
    """Ensemble model combining transformer and gradient boosting model"""

    def __init__(
        self,
        transformer_model,
        gb_model,
        text_processor,
        tokenizer,
        weights: Optional[List[float]] = None,
    ):
        self.transformer_model = transformer_model
        self.gb_model = gb_model
        self.text_processor = text_processor
        self.tokenizer = tokenizer
        weights = weights or [0.5, 0.5]
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

    def predict(self, texts: List[str], domain_idx: Optional[int] = None) -> np.ndarray:
        """Get predictions from the ensemble"""
        self.transformer_model = self.transformer_model.to("cpu")

        # get predictions from the gradient boosting model
        gb_features = self.text_processor.transform(texts)
        gb_preds = self.gb_model.predict_proba(gb_features)

        # get predictions from the transformer model
        transformer_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            transformer_outputs = self.transformer_model(**transformer_inputs)
            transformer_preds = torch.softmax(transformer_outputs.logits, dim=1).numpy()

        # combine predictions using normalized weights
        ensemble_preds = (
            self.weights[0] * transformer_preds + self.weights[1] * gb_preds
        )

        # convert probabilities to binary predictions
        return (ensemble_preds[:, 1] >= 0.5).astype(int)


def create_transformer_model(model_name: str = "distilbert-base-uncased"):
    """Initialize pre-trained transformer model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model, tokenizer
