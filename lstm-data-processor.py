import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import importlib
from transformers import AutoTokenizer

hyperparams = importlib.import_module('lstm-hyperparams')

TOKENIZER_NAME = 'emilyalsentzer/Bio_ClinicalBERT'

class MIMICDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, data_type: str, model_type: str, max_len: int = hyperparams.SEQ_LEN):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_type = data_type
        self.model_type = model_type
        self.num_samples = len(dataframe)
            
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]['text'])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        indices = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        if self.data_type == 'test': 
            return indices, attention_mask

        label = torch.tensor(self.dataframe.iloc[idx]['label'], dtype=torch.float32)
        return indices, attention_mask, label

def load_and_preprocess_data(data_paths: list[str], data_type: str, model_type: str, max_len: int = hyperparams.SEQ_LEN):
    df = pd.DataFrame()
    for path in data_paths:
        df = pd.concat([df, pd.read_csv(path)])

    df["text"] = df["text"].fillna("")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=None)

    if data_type == 'train_val':
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True, stratify=df['label']
        )
        print(f"Data loaded. Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_set = MIMICDataset(dataframe=train_df.reset_index(drop=True), tokenizer=tokenizer, max_len=max_len, data_type='train', model_type=model_type)
        val_set = MIMICDataset(dataframe=val_df.reset_index(drop=True), tokenizer=tokenizer, max_len=max_len, data_type='val', model_type=model_type)
        return train_set, val_set, tokenizer

    print(f"Data loaded for {data_type}. Number of samples: {len(df)}")
    dataset = MIMICDataset(dataframe=df, tokenizer=tokenizer, max_len=max_len, data_type=data_type, model_type=model_type)

    if data_type in ['train', 'val']:
        return dataset, tokenizer
    else:
        return dataset

def prepare_standard_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle
    )

def prepare_distributed_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        sampler=DistributedSampler(dataset)
    )