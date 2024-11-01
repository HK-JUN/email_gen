import pandas as pd

dt = pd.read_csv("/home/user3/workplace/dataset/politeness.csv")
useful_dt = dt[dt['is_useful']==1]
from datasets import Dataset, DatasetDict
train_data = useful_dt[useful_dt["split"] == 'train']
test_data = useful_dt[useful_dt["split"] == 'test']
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
dataset_dict = DatasetDict({
    'train':train_dataset,
    'test':test_dataset
}
)

from transformers import BartTokenizer, BartForConditionalGeneration
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def bart_summarize_text(text):
    inputs = bart_tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=1.5, num_beams=6, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

import pandas as pd
import numpy as np

def process_in_chunks(df, chunk_size, output_file):
    chunks = np.array_split(df, np.ceil(len(df) / chunk_size))  # 데이터셋을 청크로 나눔
    for i, chunk in enumerate(chunks):
        chunk['summary'] = chunk['txt'].apply(bart_summarize_text)
        chunk.to_csv(f'{output_file}train_part_{i}.csv', index=False)  # 각 청크를 파일로 저장
        print(f"{i} chunk completed")
df_test = pd.DataFrame(dataset_dict['train'])
chunk_size = 2000  
process_in_chunks(df_test, chunk_size, '/home/user3/workplace/dataset/train/')