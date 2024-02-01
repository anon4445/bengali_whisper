# Bengali-Whisper: A Transformer-Based Eﬀicient Framework for Automatic Bengali Speech Recognition

Despite its promising capabilities, the original Weakly-Supervised Deep Learning Acoustic Model (Whisper)-based automatic speech recognition (ASR) model exhibited limited effectiveness in recognizing Bengali, the seventh most spoken language in the world. This limitation highlights a critical need for targeted enhancements to better serve this substantial user base. This study focuses on improving the performance of the Whisper ASR for the Bengali language. Our approach involved fine-tuning the model with a learning rate 40 times smaller, coupled with the use of a minimal yet strategically augmented dataset. These interventions led to a significant 57\% decrease in error rate compared to the original Whisper model for Bengali. Further validation through bootstrap analysis underscored this improvement, revealing a substantial and statistically significant error rate reduction in the fine-tuned model compared to the pretrained Whisper Base model in ASR. This was evidenced by a completely negative confidence interval at a 99\% confidence level. The findings demonstrate the profound impact of targeted learning rate adjustments and data augmentation in enhancing the performance of transformer-based ASR models, particularly for low-resource languages, thereby expanding Whisper’s applicability and setting a new benchmark for adapting speech recognition technologies to Bengali linguistic contexts.


## Reproduce the experiment: 

Download the common voice 9 dataset for Benglai Language. 
https://commonvoice.mozilla.org/en/datasets 

Split the test, train, validation audio first, run: 
```
python data_preparation.py
```
Run augmentation: 
```
python augmentation.py 
```
Run trainer: 
```
python trainer.py 
``` 
Run evaluator: 

For Statistical significance test between two ASR models using bootstrap sampling: 
Please check this repo: 
```
https://github.com/jakariaemon/ASR-statistical-significance-using-bootstrap-sampling
``` 

