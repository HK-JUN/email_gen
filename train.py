import torch
from transformers import BartTokenizer, BartForConditionalGeneration

dataset = []


model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def preprocess(batch):
    inputs = tokenizer(batch['summary'],max_length=512,truncation=True,padding="max_length",return_tensors="pt")
    targets = tokenizer(batch['txt'],max_length=150,truncation=True,padding="max_length",return_tensors="pt")
    batch['input_ids'] = inputs['input_ids']
    batch['attention_mask'] = inputs['attention_mask']
    batch['labels'] = targets['input_ids']
    return batch

