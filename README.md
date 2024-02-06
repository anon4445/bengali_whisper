# Bengali-Whisper: A Transformer-Based Eﬀicient Framework for Automatic Bengali Speech Recognition


## Reproduce the experiment: 

Download the augmented dataset:  
[Bengali-Whisper_CV9_Augmented Huggingface](https://huggingface.co/datasets/emon-j/Bengali-Whisper_CV9_Augmented)

Run trainer: 
```
python trainer.py 
``` 
Run evaluator: 
```
python evaluate_common_voice.py
python fleurs.py
```

For Statistical significance test between two ASR models using bootstrap sampling: 
[ASR-statistical-significance](https://github.com/jakariaemon/ASR-statistical-significance-using-bootstrap-sampling)

