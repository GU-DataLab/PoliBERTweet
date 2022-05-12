# 🎊 PoliBERTweet: Language Models for Political Tweets
Transformer-based language models pre-trained on a large amount of politics-related Twitter data (83M tweets). This repo is the official resource of the following paper.
- [PoliBERTweet: A Pre-trained Language Model for Analyzing Political Content on Twitter](https://lrec2022.lrec-conf.org/en/conference-programme/accepted-papers/), LREC 2022.

## 📚 Data Sets
The data sets for the evaluation tasks presented in [our paper](https://lrec2022.lrec-conf.org/en/conference-programme/accepted-papers/) are available below.

- Poli-Test & NonPoli-Test - [Coming Soon]
- Stance Data Sets - [[Data](https://portals.mdi.georgetown.edu/public/stance-detection-KE-MLM)] [[Paper](https://aclanthology.org/2021.naacl-main.376/)] [[Github](https://github.com/GU-DataLab/stance-detection-KE-MLM)]

## 🚀 Pre-trained Models

All models are uploaded to my [Huggingface](https://huggingface.co/kornosk) 🤗 so you can load model with **just three lines of code**!!!

- [PoliBERTweet](https://huggingface.co/kornosk/polibertweet-mlm) (83M tweets) - Feel free to fine-tune this to any downstream task 🎯
- [PoliBERTweet-small](https://huggingface.co/kornosk/polibertweet-mlm-small) (5M tweets)

## ⚙️ Usage

We tested in `pytorch v1.8.1` and `transformers v4.5.1`.

Please see specific model pages above for more usage detail. Below is a sample use case.

### Example for Masked Token Prediction
```python
from transformers import AutoModel, AutoTokenizer, pipeline
import torch

# choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# select mode path here
pretrained_LM_path = "kornosk/polibertweet-mlm"

# load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModel.from_pretrained(pretrained_LM_path)

# fill mask
example = "Trump is the <mask> of USA"
fill_mask = pipeline('fill-mask', model=pretrained_LM_path, tokenizer=tokenizer)

outputs = fill_mask(example)
print(outputs)

# see embeddings
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
print(outputs)

# OR you can use this model to train on your downstream task!
# please consider citing our paper if you feel this is useful :)
```

## ✏️ Citation
If you feel our paper and resources are useful, please consider citing our work! 🙏
```bibtex
@inproceedings{kawintiranon2022polibertweet,
  title     = {PoliBERTweet: A Pre-trained Language Model for Analyzing Political Content on Twitter},
  author    = {Kawintiranon, Kornraphop and Singh, Lisa},
  booktitle = {Proceedings of the Language Resources and Evaluation Conference},
  year      = {2022},
  publisher = {European Language Resources Association}
}
```
