import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import unicodedata
from tqdm import tqdm
from jiwer import wer
import wandb

class SpeechRecognitionEvaluator:
    def __init__(self, model_id, dataset_name, dataset_config, device="cuda", torch_dtype=torch.float16):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=1,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.dataset = load_dataset(dataset_name, dataset_config, split='test')

    @staticmethod
    def remove_punctuation(text):
        bn_punctuation = u"।,"
        return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'P' and ch not in bn_punctuation)

    def evaluate(self):
        wandb.init(project="bnasr_evaluation_flerus")
        evaluation_table = wandb.Table(columns=["Original Transcription", "Predicted Transcription", "WER"])
        total_wer = 0
        total_samples = 0

        for sample in tqdm(self.dataset, desc="Processing files"):
            audio = sample["audio"]
            original_transcription = self.remove_punctuation(sample["transcription"])
            result = self.pipe(audio, generate_kwargs={"task": "transcribe", "language": "bengali"})
            cleaned_result = self.remove_punctuation(result["text"])
            sample_wer = wer(original_transcription, cleaned_result)
            evaluation_table.add_data(original_transcription, cleaned_result, sample_wer)
            total_wer += sample_wer
            total_samples += 1

        average_wer = total_wer / total_samples if total_samples > 0 else 0
        wandb.log({"Average WER": average_wer, "Evaluation Table": evaluation_table})
        wandb.finish()

        print(f"Average WER: {average_wer}")

# Example usage
model_id = "openai/whisper-base"  # or set fine_tuned model folder 
dataset_name = "google/fleurs"
dataset_config = "bn_in"
speech_recognition_evaluator = SpeechRecognitionEvaluator(model_id, dataset_name, dataset_config)
speech_recognition_evaluator.evaluate()
