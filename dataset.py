from os import path

import pandas as pd

import torch
from torch.utils import data
from torch.nn.utils import rnn

import torchtext

from sklearn import model_selection


DATA_DIR = 'data'


class Dataset(data.Dataset):
    def __init__(self, content, tokenizer, vocab):
        self.content = content
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        text, label = self.content[idx]
        tokens = self.tokenizer(text)
        tokens_ids = self.vocab(tokens)
        return torch.tensor(tokens_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)

    def collate_fn(self, batch):
        tokens_ids, labels = zip(*batch)
        tokens_ids = rnn.pad_sequence(tokens_ids, batch_first=True, padding_value=self.vocab['<pad>'])
        # torch.tensor() doesn't accept tuples.
        labels = [list(label) for label in labels]
        return tokens_ids, torch.tensor(labels, dtype=torch.float)


def get_dataset_path(filename):
    return path.join(DATA_DIR, filename)


def build_vocab(texts, tokenizer):
    def iter_tokens():
        for text in texts:
            yield tokenizer(text)

    vocab = torchtext.vocab.build_vocab_from_iterator(iter_tokens(), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def load_default_datasets_and_vocab():
    true_path = get_dataset_path('true.csv')
    fake_path = get_dataset_path('fake.csv')

    text_column = 'text'
    title_column = 'title'
    subject_column = 'subject'
    date_column = 'date'

    label_column = 'label'

    test_weight = 0.2
    train_test_split_seed = 42

    tokenizer_model = 'basic_english'
    language = 'en'

    true_dataframe = pd.read_csv(true_path)
    fake_dataframe = pd.read_csv(fake_path)

    true_dataframe[label_column] = [(1, 0)] * len(true_dataframe)
    fake_dataframe[label_column] = [(0, 1)] * len(fake_dataframe)

    dataframe = pd.concat([true_dataframe, fake_dataframe], ignore_index=True)
    dataframe.drop([title_column, subject_column, date_column], axis='columns', inplace=True)
    # Remove blank texts (which are unwillingly there).
    dataframe = dataframe[dataframe[text_column].str.strip().astype(bool)]

    train_dataframe, test_dataframe = model_selection.train_test_split(
        dataframe,
        test_size=test_weight,
        random_state=train_test_split_seed)

    tokenizer = torchtext.data.utils.get_tokenizer(tokenizer_model, language)

    # Currently the vocab is based purely on the training and testing datasets.
    vocab = build_vocab(dataframe[text_column], tokenizer)

    train_dataset = Dataset(tuple(zip(train_dataframe[text_column], train_dataframe[label_column])), tokenizer, vocab)
    test_dataset = Dataset(tuple(zip(test_dataframe[text_column], test_dataframe[label_column])), tokenizer, vocab)

    return train_dataset, test_dataset, vocab


def load_datasets_and_vocab():
    return load_default_datasets_and_vocab()
