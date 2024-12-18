# Bengali-Whisper:Efficient Adaptation of Trasnformer Architecture for Bengali Speech Recognition

This repository contains the code and resources to reproduce the experiments presented in the paper "Bengali-Whisper: Efficient Adaptation of Trasnformer Architecture for Bengali
Speech Recognition".

## Reproduce the experiment:

### Download the augmented dataset:

*   [Augmented CV-BN](https://huggingface.co/datasets/emon-j/Bengali-Whisper_CV9_Augmented)

### Fine-tuned models (Decoder Only):

*   [FineTuned-Base BN-DecOnly](https://huggingface.co/emon-j/Bengali-Whsiper)
*   [FineTuned-Small BN-DecOnly](https://huggingface.co/emon-j/Bengali-Whsiper)
*   [FineTuned-Large-v2 BN-DecOnly](https://huggingface.co/emon-j/Bengali-Whsiper)

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install DeepSpeed:**
    DeepSpeed installation requires specific CUDA toolkit versions. Please follow the official DeepSpeed installation instructions for detailed guidance:

    *   **Official Installation Guide:** [https://www.deepspeed.ai/tutorials/advanced-install/](https://www.deepspeed.ai/tutorials/advanced-install/)

    Here's a basic example using `pip`:
    ```bash
    pip install deepspeed
    ```

3.  **Train the model:**
    ```bash
    deepspeed trainer.py 
    ```

4.  **Evaluate the model:**
    ```bash
    python evaluate_common_voice.py
    python fleurs.py
    ```

## Methodology

This study focuses on fine-tuning the pre-trained Whisper model for Bengali speech recognition. The key steps involved are:

1.  **Data preparation:**  The Common Voice Corpus 9.0 Bengali dataset is used for training and evaluation.
2.  **Data augmentation:**  The training data is augmented using SpecAugment, dithering noise addition, and mixed augmentation to improve model robustness and generalization.
3.  **Fine-tuning:**  The Whisper model is fine-tuned using both full fine-tuning and decoder-only strategies.
4.  **Hyperparameter optimization:**  Bayesian hyperparameter optimization is performed to determine the optimal learning rate.
5.  **Evaluation:**  The fine-tuned models are evaluated on the Common Voice and Fleurs datasets using Word Error Rate (WER) and Character Error Rate (CER) metrics.
![asr](https://github.com/user-attachments/assets/95746114-f8ca-48bb-b040-72b2d37d814f)

## Whisper Architecture

Whisper is a Transformer-based encoder-decoder model. The encoder processes audio input to extract latent representations, while the decoder generates text output based on these representations.

*   **Encoder:**  The encoder consists of convolutional layers for feature extraction, positional encoding, and multiple stacked residual attention blocks. Each attention block contains a self-attention mechanism and a feed-forward network.
*   **Decoder:**  The decoder utilizes pre-trained token embeddings, learned positional embeddings, and residual attention blocks with self-attention and encoder-decoder cross-attention mechanisms. It also employs special tokens for multi-task operations.

![image](https://github.com/user-attachments/assets/109bc430-4194-4293-a9f2-012f555185d2)

## Results 

![image](https://github.com/user-attachments/assets/f1d25728-16cc-4c3c-af82-794eac7723f3)

## Transcription Comparison 
![image](https://github.com/user-attachments/assets/42839284-1cef-485a-8b65-bdf763bd9d57)


## Statistical Significance Test

For conducting a statistical significance test between two ASR models using bootstrap sampling, refer to the following repository:

*   [ASR-statistical-significance](https://github.com/jakariaemon/ASR-statistical-significance-using-bootstrap-sampling)

## Citation

If you find this code or the corresponding paper useful for your research, please cite:
```
Upcoming 
``` 
