import os
import shutil
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    AutoConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
)
import wandb

@dataclass
class CustomDataCollatorWithPadding:
    """
    A data collator that dynamically pads the inputs and labels.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

class WhisperTrainer:
    def __init__(self, model_name: str, dataset_name: str, language: str = "Bengali", task: str = "transcribe"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.language = language
        self.task = task
        self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
        self.model = self._load_model()
        self.data_collator = CustomDataCollatorWithPadding(processor=self.processor)

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.model_name, use_cache=False)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name, config=config)
        return model

    def _prepare_dataset(self, batch):
        audio = batch['path']
        batch["input_features"] = self.processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    def _modify_path(self, batch):
        batch['path'] = os.path.join("bn/clips", batch['path'])
        return batch

    def _compute_metrics(self, pred):
        metric = evaluate.load("wer")
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        wandb.log({"wer": wer}, commit=False)
        return {"wer": wer}

    def train(self, learning_rate=2.413e-05, num_train_epochs=10):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./whisper-finetuned-test",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            warmup_steps=50,
            num_train_epochs=num_train_epochs,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="epoch",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=228,
            logging_steps=25,
            report_to=["wandb"],
            load_best_model_at_end=False,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            learning_rate=learning_rate,
        )

        dataset = DatasetDict()
        dataset = load_dataset(self.dataset_name)
        dataset = dataset.map(self._modify_path)
        dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
        dataset = dataset.map(self._prepare_dataset, num_proc=1)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        trainer.train()
        trainer.save_model(training_args.output_dir)


whisper_trainer = WhisperTrainer(
    model_name="openai/whisper-base",
    dataset_name="your_dataset_name_here"  
)
whisper_trainer.train()
