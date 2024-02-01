from datasets import load_dataset
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import unicodedata
import os
from tqdm import tqdm
from jiwer import wer
import wandb

class ASREvaluator:
    def __init__(self, model_id, dataset_name, dataset_split, project_name):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.project_name = project_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer,
                             feature_extractor=self.processor.feature_extractor, max_new_tokens=128,
                             torch_dtype=self.torch_dtype, device=self.device)
        self.dataset = load_dataset(dataset_name, split=dataset_split)

    @staticmethod
    def remove_punctuation(input_text):
        punctuation_marks = u"।,"  # Bengali full stop and additional punctuation to remove
        return ''.join(ch for ch in input_text if unicodedata.category(ch)[0] != 'P' and ch not in punctuation_marks)

    def evaluate(self):
        wandb.init(project=self.project_name)
        total_wer = 0
        total_samples = 0
        evaluation_table = wandb.Table(columns=["Filepath", "Original Transcription", "Predicted Transcription", "WER"])

        for sample in tqdm(self.dataset, desc="Processing files"):
            filepath = os.path.join("bn/clips", sample['path'])
            original_transcription = self.remove_punctuation(sample['sentence'])
            transcription = self.pipe(filepath)
            sample_wer = wer(original_transcription, transcription['text'])
            evaluation_table.add_data(filepath, original_transcription, transcription['text'], sample_wer)
            total_wer += sample_wer
            total_samples += 1

        average_wer = total_wer / total_samples if total_samples > 0 else 0
        wandb.log({"Average WER": average_wer, "Evaluation Table": evaluation_table})
        wandb.finish()

        print(f"Average WER: {average_wer}")

# Example usage:
model_id = "whisper-base-fn"
dataset_name = "bn"
dataset_split = "test"
project_name = "bnasr_evaluation_cv9"
asr_evaluator = ASREvaluator(model_id, dataset_name, dataset_split, project_name)
asr_evaluator.evaluate()
