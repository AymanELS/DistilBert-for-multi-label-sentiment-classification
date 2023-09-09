from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch



## load dataset
dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english", split=['train[:4000]', 'test[:1000]', 'validation'])
## split training into training and validation
dataset_train, dataset_test, dataset_val = dataset[0], dataset[1], dataset[2]

# print(dataset_train[0])

## encode idx and labels
labels = list(dataset_train.features.keys())[2:]  #get labels (remove 'ID' and 'Tweet')
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}


##Data preprocessing
labels_batch = {k: dataset_train[0][k] for k in dataset_train[0].keys() if k in labels}
# print(dataset_train[0]['anticipation'])
labels_batch = {k: dataset_train[0][k] for k in labels}
# print(label_batch)



##Data preprocessing
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_data(data):
  # encoding with truncation based on max length
  text = data['Tweet']
  encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128)
  # add labels
  labels_batch = {k: data[k] for k in labels}
  # init label matrix
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill label matrix
  for id, label in enumerate(labels):
    labels_matrix[:, id] = labels_batch[label]
  encoding['labels'] = labels_matrix.tolist()
  return encoding


encoded_dataset_train = dataset_train.map(preprocess_data, batched=True, remove_columns=dataset_train.column_names)
encoded_dataset_val = dataset_val.map(preprocess_data, batched=True, remove_columns=dataset_train.column_names)
encoded_dataset_test = dataset_test.map(preprocess_data, batched=True, remove_columns=dataset_train.column_names)

# example = encoded_dataset_train[0]
# print(example.keys())
# print(tokenizer.decode(example['input_ids']))
# print([id2label[id] for id, label in enumerate(example['labels']) if label==1.0 ])
encoded_dataset_train.set_format("torch")
encoded_dataset_val.set_format("torch")
encoded_dataset_test.set_format("torch")

## Load predefined DistilBert model for finetuning. Set the task to multi-class classification

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', problem_type='multi_label_classification', num_labels=len(labels), id2label=id2label, label2id=label2id)


## setup the huggingface trainer
batch_size = 8
metric_name = "f1"

args = TrainingArguments(
    "distilbert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true, y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds,  labels=p.label_ids)
    return result

# print(len(encoded_dataset_train))
# print(len(encoded_dataset_test))
# print(len(encoded_dataset_val))

# outputs = model(input_ids=encoded_dataset_train['input_ids'][0].unsqueeze(0), attention_mask=encoded_dataset_train['attention_mask'][0].unsqueeze(0), labels=encoded_dataset_train[0]['labels'].unsqueeze(0))
# print(outputs)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# train model
trainer.train()

# inference
def test_performance(output):
  logits = output.logits
  sigmoid = torch.nn.Sigmoid()
  probs = sigmoid(logits.squeeze().cpu())
  predictions = np.zeros(probs.shape)
  predictions[np.where(probs >= 0.5)] = 1
  # turn predicted id's into actual label names
  predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
  print(predicted_labels)

text = "We had a great trip last week, hopefully we can do it again soon"

encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
output = trainer.model(**encoding)
test_performance(output)
# >>> ['joy', 'optimism']
