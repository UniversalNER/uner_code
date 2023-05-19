# -*- coding: utf-8 -*-
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import os
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from prepare_data import read_uner
import evaluate
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"
github_token = os.environ["GITHUB_TOKEN"]
data_url = "https://raw.githubusercontent.com/UniversalNER/UNER_English-EWT/master/en_ewt-ud-{split}.iob2"
LABEL_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

# download from the data_url to a local file
def download_file(data_url, output_file):
  # include authorization in requests
  with requests.get(data_url, headers={'Authorization': f'Token {github_token}'}, stream=True) as r:
      r.raise_for_status()
      with open(output_file, 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192): 
              f.write(chunk)

def compute_metrics(predictions_labels):
  metric = evaluate.load("seqeval")
  predictions, labels = predictions_labels
  predictions = np.argmax(predictions, axis=2)

  # Remove ignored index (special tokens)
  true_predictions = [
      [LABEL_NAMES[p] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]
  true_labels = [
      [LABEL_NAMES[l] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]

  results = metric.compute(predictions=true_predictions, references=true_labels)
  flattened_results = {
      "overall_precision": results["overall_precision"],
      "overall_recall": results["overall_recall"],
      "overall_f1": results["overall_f1"],
      "overall_accuracy": results["overall_accuracy"],
  }
  for k in results.keys():
    if(k not in flattened_results.keys()):
      flattened_results[k+"_f1"]=results[k]["f1"]

  return flattened_results

class UNERConfig:
  # a bunch of stuff...
  dataset = "en_ewt-ud"
  model_name = "bert-base-multilingual-cased"
  num_epochs = 10
  dev = False

class UNERTrainer():

  def __init__(self, config: UNERConfig):
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
    self.tokenized_dataset = self.setup_data()
    # this line has to happen after the above one
    self.trainer = self.setup_trainer()


  def setup_data(self):
    download_file(data_url.format(split="train", github_token=github_token), f'{self.config.dataset}-train.iob2')
    download_file(data_url.format(split="dev", github_token=github_token), f'{self.config.dataset}-dev.iob2')
    download_file(data_url.format(split="test", github_token=github_token), f'{self.config.dataset}-test.iob2')

    train_path = f"{self.config.dataset}-train.iob2"
    dev_path = f"{self.config.dataset}-dev.iob2"
    test_path = f"{self.config.dataset}-test.iob2"

    training_dataset = read_uner(train_path)
    validation_dataset = read_uner(dev_path)
    test_dataset = read_uner(test_path)

    if self.config.dev:
      subsample_num = 100
      training_dataset = Dataset.from_dict(training_dataset[:subsample_num])
      validation_dataset = Dataset.from_dict(validation_dataset[:subsample_num])
      test_dataset = Dataset.from_dict(test_dataset[:subsample_num])

    print(training_dataset)
    print(validation_dataset)

    # create a new dataset from the training validation sets above
    #dataset = Dataset.from_dict({"train": training_dataset, "validation": validation_dataset})

    # create a new DatasetDict from the training and validation sets above
    dataset = DatasetDict({"train": training_dataset, "validation": validation_dataset, "test": test_dataset})

    # FIXME: this is probably broken! You shouldn't copy B-ORG, for example.

    # Get the values for input_ids, token_type_ids, attention_mask
    def tokenize_adjust_labels(all_samples_per_split):
      tokenized_samples = self.tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True)
      # tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
      # so the new keys [input_ids, labels (after adjustment)]
      # can be added to the datasets dict for each train test validation split
      total_adjusted_labels = []
      # this is batch size.
      batch_size = len(tokenized_samples["input_ids"])

      # Oh, i think I get it now. The tokenizer splits stuff up
      # but we need to adujust the labels to match the tokens

      for k in range(0, batch_size):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_labels = all_samples_per_split["ner_tags"][k]
        existing_label_ids = [LABEL_NAMES.index(l) for l in existing_labels]

        i = -1
        adjusted_label_ids = []
      
        for wid in word_ids_list:
          if wid is None:
            # ??? what does -100 mean? I guess it's some kind of dummy label
            adjusted_label_ids.append(-100)
          elif wid != prev_wid:
            i = i + 1
            adjusted_label_ids.append(existing_label_ids[i])
            prev_wid = wid
          else:
            # label_name = label_names[existing_label_ids[i]]
            adjusted_label_ids.append(existing_label_ids[i])
            
        total_adjusted_labels.append(adjusted_label_ids)

      tokenized_samples["labels"] = total_adjusted_labels
      return tokenized_samples

    return dataset.map(tokenize_adjust_labels, batched=True)

  def setup_trainer(self):
    data_collator = DataCollatorForTokenClassification(self.tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(self.config.model_name, num_labels=len(LABEL_NAMES))
    training_args = TrainingArguments(
        output_dir="./universal-ner",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1 if self.config.dev else self.config.num_epochs,
        weight_decay=0.01,
        logging_steps = 500,
        run_name = "uner_train",
        save_strategy='steps',
        push_to_hub = False,
        hub_private_repo = True,
        load_best_model_at_end = True,
        disable_tqdm=True
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=self.tokenized_dataset["train"],
        eval_dataset=self.tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=self.tokenizer,
        compute_metrics=compute_metrics
    )

  def train(self):
    self.trainer.train()

  def evaluate(self):
    predictions = self.trainer.predict(self.tokenized_dataset["test"])
    print(predictions.metrics)
    return predictions.metrics["test_overall_accuracy"], predictions.metrics["test_overall_f1"]


if __name__ == "__main__":
  cfg = UNERConfig()
  cfg.dev = True
  trainer = UNERTrainer(cfg)
  trainer.train()
  trainer.evaluate()