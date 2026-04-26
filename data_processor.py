import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import nltk
from nltk.tokenize import wordpunct_tokenize
import string
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import math

def preprocess_text(text : str) -> list:
    """
    Clean and convert a text string into a list of tokens.
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Use a tokenizer that does not require external punkt resources
        tokens = wordpunct_tokenize(text)
        return tokens
    return []

class Vocabulary:
    """
    Build a vocabulary from the word count
    """
    def __init__(self, max_size : int):
        self.max_size = max_size
        # Add <cls> token for transformer classification
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<cls>"}
        self.word_count = {}
        self.size = 3  # Start with pad, unk, and cls tokens
        
    def add_word(self, word : str):
        self.word_count[word] = self.word_count.get(word, 0) + 1
            
    def build_vocab(self):
        sorted_words = sorted(self.word_count.items(), key=lambda item: item[1], reverse=True)
        for word, count in sorted_words:
            if self.size < self.max_size:
                self.word2idx[word] = self.size
                self.idx2word[self.size] = word
                self.size += 1
            else:
                break

        print(f"Vocabulary built with {self.size} words (including special tokens).")

    def text_to_indices(self, tokens : list, max_len : int, model_type : str = 'lstm') -> list[int]:
        """
        Convert a list of tokens into a list of token ids.
        """
        if  model_type == 'transformer':
            tokens =  ["<cls>"] + tokens[:-1]
        
        if tokens.__len__() > max_len:
            tokens = tokens[:max_len]
        elif tokens.__len__() < max_len:
            tokens = tokens + ["<pad>"] * (max_len - tokens.__len__())
        
        indices = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
        return indices

class MIMICDataset(Dataset):
    """
    A dataset for the MIMIC-III dataset
    """
    def __init__(self, dataframe : pd.DataFrame, vocabulary : Vocabulary, max_len : int, is_training : bool = True, model_type : str = 'lstm'):
        self.dataframe = dataframe
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.is_training = is_training
        self.model_type = model_type
        self.num_samples = len(dataframe)
            
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        text = self.dataframe.iloc[idx]['text']  # Text data of the sample
        label = self.dataframe.iloc[idx]['label']  # Sample label
        tokens = preprocess_text(text)  # Convert text data to list of tokens
        indices = self.vocabulary.text_to_indices(tokens, self.max_len, model_type=self.model_type)  # Convert to id list

        if self.model_type == 'transformer':
            attention_mask = [1 if index != self.vocabulary.word2idx["<pad>"] else 0 for index in indices]
            return torch.tensor(indices), torch.tensor(attention_mask), torch.tensor([label])  # (features, mask, label) for transformer
        elif self.model_type == 'lstm' or self.model_type == 'rnn':
            return torch.tensor(indices), torch.tensor([label])  # (features, label) for LSTM
        else:
            raise ValueError("Invalid model type. Choose 'lstm' or 'transformer' or 'rnn'.")

def load_and_preprocess_data(data_path : str, data_type : str = 'train', model_type : str = 'lstm', shared_vocab : Vocabulary | None = None, max_vocab_size : int = 10000, batch_size : int = 32, max_len : int = 500) -> tuple[DataLoader, DataLoader, Vocabulary] | tuple[DataLoader, Vocabulary] | DataLoader:
    """
    Load and preprocess the MIMIC-III dataset
    
    Args:
        data_path: Path to the data files
        data_type: Type of data to load ('train' or 'test' or 'train_val')
        model_type: Type of model ('lstm' or 'transformer' or 'rnn')
        shared_vocab: Optional vocabulary to use (for test data)
        max_vocab_size: Maximum size of the vocabulary
        batch_size: Batch size for the DataLoader
        max_len: Maximum length of the input sequences
    Returns:
        data_loader: DataLoader for the specified data type
        vocab: Vocabulary object (only returned for train data)
    """

    df = pd.read_csv(data_path)

    if shared_vocab is None:
        vocab = Vocabulary(max_size=max_vocab_size)

        # Generate vocabulary from all the text column data in the dataset
        for text in tqdm(df['text']):
            tokens = preprocess_text(text)
            for token in tokens:
                vocab.add_word(token)
        vocab.build_vocab()
    else:
        vocab = shared_vocab  # Use provided vocabulary

    if data_type == 'train_val':
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=df['label'],
        )
        print(
            f"Data loaded and preprocessed for mixed data."
            f"Train samples: {len(train_df)}, Val samples: {len(val_df)}"
        )

        train_loader = DataLoader(MIMICDataset(dataframe=train_df.reset_index(drop=True), vocabulary=vocab, max_len=max_len, is_training=True, model_type=model_type), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(MIMICDataset(dataframe=val_df.reset_index(drop=True), vocabulary=vocab, max_len=max_len, is_training=False, model_type=model_type), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, vocab

    print(f"Data loaded and preprocessed for {data_type} data. Number of samples: {len(df)}")
    dataset = MIMICDataset(dataframe=df, vocabulary=vocab, max_len=max_len, is_training=(data_type=='train'), model_type=model_type)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(data_type=='train'))

    print(f"DataLoader created for {data_type} data with batch size {batch_size}.")

    if data_type == 'train':
        return data_loader, vocab
    else:
        return data_loader
