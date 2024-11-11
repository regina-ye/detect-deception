from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)
import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from typing import Tuple
import os
import warnings
from process_data import DataLoader
from all_models import (
    FastTextProcessor,
    EnsembleModel,
    create_transformer_model,
)
import pandas as pd
import numpy as np


def compute_metrics(eval_pred):
    """Compute the accuracy, precision, recall, and F1 score."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class CustomTrainer(Trainer):
    """
    A custom Trainer class so that we can use a class-weighted loss.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss function using class weights"""
        labels = inputs["labels"].to(self.args.device)
        outputs = model(**inputs)
        logits = outputs["logits"]

        self.class_weights = self.class_weights.to(self.args.device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)

        # return loss and potentially the outputs
        return (loss, outputs) if return_outputs else loss


class ModelTrainer:
    """Trains and evaluates models"""

    def __init__(self, save_dir: str = "models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_transformer_data(self, df: pd.DataFrame) -> Dataset:
        """Convert pandas DataFrame to HuggingFace Dataset"""
        return Dataset.from_dict(
            {"text": df["text"].tolist(), "label": df["label"].tolist()}
        )

    def train_transformer(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame
    ) -> Tuple[torch.nn.Module, object]:
        """
        Train the transformer model with class weighting for
        imbalanced classes
        """
        train_dataset = self.prepare_transformer_data(train_data)
        val_dataset = self.prepare_transformer_data(val_data)

        # compute class weights
        label_counts = train_data["label"].value_counts()
        total_samples = len(train_data)
        class_weights = torch.tensor(
            [total_samples / label_counts[0], total_samples / label_counts[1]],
            dtype=torch.float32,
        ).to(self.device)

        model, tokenizer = create_transformer_model()
        model = model.to(self.device)

        # tokenize datasets
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        val_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # use CustomTrainer with the custom loss
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )

        trainer.train()
        return model, tokenizer

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        unbalanced: bool,
    ) -> LGBMClassifier:
        """Train the gradient boosting model using LGBMClassifier"""
        
        warnings.filterwarnings("ignore", category=UserWarning, 
                                module="lightgbm")

        model = LGBMClassifier(
            objective="binary",
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=500,
            is_unbalance=unbalanced,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
        )
        return model


def evaluate_and_print(
    model, test_data, model_name, text_processor=None, tokenizer=None
):
    """
    Evaluate model and print results.
    """
    print(f"Evaluating {model_name} model...")

    if tokenizer:
        inputs = tokenizer(
            test_data["text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(torch.device("cpu"))
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    elif text_processor:
        processed_text = text_processor.transform(test_data["text"])
        predictions = model.predict(processed_text)
    else:
        predictions = model.predict(test_data["text"].tolist())

    print(f"predictions for {model_name}: {predictions[:10]}")

    labels = test_data["label"].tolist()

    # calculate metrics
    metrics = {
        "f1": f1_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions),
        "report": classification_report(labels, predictions),
    }

    # print results
    print(metrics["report"])
    return metrics


def main():
    """Train and evaluate all models"""

    # which models to train
    run_transformer = True
    run_gb = True
    run_ensemble = True

    # initialize
    loader = DataLoader()
    trainer = ModelTrainer()
    data = loader.load_saved_data()

    if not data:
        print("No data. Must run data_processing.py first")
        return

    device = torch.device("cpu")
    torch.set_num_threads(4)

    results = {}
    for domain in data.keys():
        print(f"\ntraining models for domain: {domain}")

        # get domain data
        domain_data = data[domain]
        if not all(split in domain_data for split in ["train", "validation", "test"]):
            print(f"skipping {domain} - missing required splits")
            continue

        train_data = domain_data["train"]
        val_data = domain_data["validation"]
        test_data = domain_data["test"]

        # initialize text processor
        text_processor = FastTextProcessor()
        X_train_tfidf = text_processor.fit_transform(train_data["text"])
        X_val_tfidf = text_processor.transform(val_data["text"])

        try:
            unbalanced = True
            if domain != "fake_news" and domain != "product_reviews":
                unbalanced = False

            # train transformer
            if run_transformer:
                print(f"training transformer model for {domain}...")
                transformer_model, tokenizer = trainer.train_transformer(
                    train_data, val_data
                )
                transformer_model = transformer_model.to(device)

                # evaluate transformer
                transformer_metrics = evaluate_and_print(
                    transformer_model,
                    test_data,
                    "Transformer",
                    tokenizer=tokenizer,
                )
                results[f"{domain}_transformer"] = transformer_metrics
            else:
                transformer_model, tokenizer = None, None

            # train gradient boosting model
            if run_gb:
                print(f"training gradient boosting model for {domain}...")
                gb_model = trainer.train_gradient_boosting(
                    X_train_tfidf,
                    train_data["label"].values,
                    X_val_tfidf,
                    val_data["label"].values,
                    unbalanced,
                )

                # evaluate gradient boosting
                gb_metrics = evaluate_and_print(
                    gb_model,
                    test_data,
                    "Gradient Boosting",
                    text_processor=text_processor,
                )
                results[f"{domain}_gradient_boosting"] = gb_metrics
            else:
                gb_model = None

            # create ensemble model
            if run_ensemble and transformer_model and gb_model:
                print(f"creating ensemble model for {domain}...")
                ensemble = EnsembleModel(
                    transformer_model=transformer_model,
                    gb_model=gb_model,
                    text_processor=text_processor,
                    tokenizer=tokenizer,
                )

                # evaluate ensemble
                ensemble_metrics = evaluate_and_print(ensemble, test_data, "Ensemble")
                results[f"{domain}_ensemble"] = ensemble_metrics

        except Exception as e:
            print(f"error training models for {domain}: {str(e)}")
            continue

    return results


if __name__ == "__main__":
    main()
