# DistilBert-for-multi-label-sentiment-classification
Fine-tuning DistilBert for multi-label sentiment classification on sem_eval_2018_task_1

## Introduction
The BERT (Bidirectional Encoder Representations from Transformers) model was first published in 2018, which brought about a revolution in NLP. It is a mult-layer bidirectional transformer which encodes text data based on its left and right context. It is first pre-trained on a large corpus of text by using a Masked Language Model pre-training objective and 'next sentence prediction' task. Then the model is fine-tuned for various NLP tasks like sentiment analysis, named entity recognition, and text classification [\[1\]](https://arxiv.org/pdf/1810.04805.pdf).

![alt text](https://github.com/AymanELS/DistilBert-for-multi-label-sentiment-classification/blob/main/Bert.png)
Source: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

DistilBERT, as the name imply, is a distilled version of BERT, which was introduced by HuggingFace. It has 40% less parameters than the base version of BERT, runs 60% faster, and retains 97% of its language understanding capabilities.
This is achieved with knowledge distillation, which is a compression technique where one model is trained to reproduce the performance a larger model. This technique is particularly useful in the case of Large Language Models (LLM) that may have billions of parameters, making them less portable and slows inference time [\[2\]](https://arxiv.org/pdf/1910.01108.pdf)


## Dataset
In this project we use the SemEval-2018 Task 1: Affect in Tweet, which contains a list of tweets in different language with the goal of predicting the sentiment of the writer of the tweet. We use the tweets in written in the English language. The list of possible sentiments include 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', and 'trust'.
Source: [\[3\]](https://aclanthology.org/S18-1001.pdf)

## Result
Custom input text: "We had a great trip last, hopefully we can do it again soon".\\
Model output: ['Joy', 'Optimism']
