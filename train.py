import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset,DatasetDict
import pandas as pd
#need to define dataset fraction into the dataset
train_data = pd.read_csv('/home/user3/workplace/dataset/train/combined_train.csv')
test_data = pd.read_csv('/home/user3/workplace/dataset/test/combined_test.csv')
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
dataset_dict = DatasetDict({
    'train':train_dataset,
    'test':test_dataset
})


model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def preprocess(batch):
    input_text = f"politeness: {batch['p_tag']} / email content:{batch['summary']}" #fix! - not a single object. batch.
    inputs = tokenizer(input_text,max_length=512,truncation=True,padding="max_length",return_tensors="pt")
    targets = tokenizer(batch['txt'],max_length=150,truncation=True,padding="max_length",return_tensors="pt")
    batch['input_ids'] = inputs['input_ids']
    batch['attention_mask'] = inputs['attention_mask']
    batch['labels'] = targets['input_ids']
    return batch

train_dataset = train_dataset.map(preprocess,batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

train_dataset.set_format(type='torch',columns=['input_ids','attention_mask','labels'])
test_dataset.set_format(type='torch',columns=['input_ids','attention_mask','labels'])

training_args = TrainingArguments(
    output_dir="../results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_training_batch_size=2,
    per_device_eval_batch_size=2,
    num_training_epoch=3,
    weight_decay=0.01,
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset = test_dataset
)

trainer.train()