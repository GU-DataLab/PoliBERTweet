# PoliBERTweet
A transformer-based language model trained on politics-related Twitter data. This repo is the official resource of the paper "PoliBERTweet: A Pre-trained Language Model for Analyzing Political Content on Twitter", LREC 2022. 🚀

## Pre-trained Models

All models are uploaded to my [Huggingface](https://huggingface.co/kornosk) 🤗 so you can load model with **just three lines of code**!!!

- [xxx](https://huggingface.co/kornosk/xxx) - Feel free to fine-tune this to any downstream task 🎯

## Usage

We tested in `pytorch v1.8.1` and `transformers v4.5.1`.

Please see specific model pages above for more usage detail. Below is a sample use case. 

### 1. Choose and load model for stance detection

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# select mode path here
# see more at https://huggingface.co/kornosk
pretrained_LM_path = "kornosk/xxx"

# load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)
```

### 2. Get a prediction (see more in `sample_predict.py`)
```python
id2label = {
    0: "AGAINST",
    1: "FAVOR",
    2: "NONE"
}

##### Prediction Favor #####
sentence = "Go Go Biden!!!"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

print("Sentence:", sentence)
print("Prediction:", id2label[np.argmax(predicted_probability)])
print("Against:", predicted_probability[0])
print("Favor:", predicted_probability[1])
print("Neutral:", predicted_probability[2])

# please consider citing our paper if you feel this is useful :)
```

## Citation
If you feel our paper and resources are useful, please consider citing our work! 🙏
```bibtex
@inproceedings{kawintiranon2022polibertweet,
    title={PoliBERTweet: A Pre-trained Language Model for Analyzing Political Content on Twitter},
    author={Kawintiranon, Kornraphop and Singh, Lisa},
    booktitle={xxx},
    year={2022},
    publisher={Association for Computational Linguistics},
    url={xxx}
}
```
