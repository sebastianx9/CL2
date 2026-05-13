"""
I am using byte-level tokenisation with a +4 shift to reserve indices 0-3 for special tokens:
    PAD=0, BOS=1, EOS=2, SEP=3
All content bytes will occupy indices 4-259, giving a vocabulary size of 260.
"""

import urllib.request
import os
import torch
from torch.utils.data import Dataset, DataLoader

PAD_IDX   = 0
BOS_IDX   = 1
EOS_IDX   = 2
SEP_IDX   = 3
VOCAB_SIZE = 260     # 256 byte values + 4 special tokens (PAD, BOS, EOS, SEP)

BASE_URL = "https://raw.githubusercontent.com/sigmorphon/2023InflectionST/main/part1/data/"


def download_data(data_dir='.', lang='eng'):
    for split in ['trn', 'dev', 'tst']:
        fname = f'{lang}.{split}'
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(BASE_URL + fname, path)


def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                lemma    = parts[0]
                features = parts[1]
                form     = parts[2] if len(parts) > 2 else ''
                data.append((lemma, features, form))
    return data


def encode_src(lemma, features):
    # encode lemma and features separately, insert SEP_IDX (3) between them
    lemma_tokens   = [b + 4 for b in lemma.encode('utf-8')]
    feature_tokens = [b + 4 for b in features.encode('utf-8')]
    return lemma_tokens + [SEP_IDX] + feature_tokens


def encode_tgt(form, add_bos=True, add_eos=True):
    tokens = [b + 4 for b in form.encode('utf-8')]
    if add_bos:
        tokens = [BOS_IDX] + tokens
    if add_eos:
        tokens = tokens + [EOS_IDX]
    return tokens


def decode_ids(ids):
    # convert token ids back to string; stop at EOS
    result = []
    for tok in ids:
        if tok == EOS_IDX:
            break
        if tok >= 4:
            result.append(tok - 4)
    return bytes(result).decode('utf-8', errors='replace')


def pad_batch(seqs, pad_idx=PAD_IDX):
    # sequences in a batch have different lengths — pad all to the longest
    max_len = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_idx] * (max_len - len(s)) for s in seqs])


class InflectionDataset(Dataset):
    def __init__(self, data):
        self.data = [(l, f, t) for l, f, t in data if t]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lemma, features, form = self.data[idx]
        src = encode_src(lemma, features)
        tgt = encode_tgt(form, add_bos=True, add_eos=True)
        return src, tgt, lemma, features, form


def collate_fn(batch):
    # DataLoader's default collate requires equal lengths — override with custom padding
    srcs, tgts, lemmas, feats, forms = zip(*batch)
    return pad_batch(list(srcs)), pad_batch(list(tgts)), lemmas, feats, forms


def get_dataloaders(data_dir='.', batch_size=64, lang='eng'):
    download_data(data_dir, lang)
    train_data = load_data(os.path.join(data_dir, f'{lang}.trn'))
    dev_data   = load_data(os.path.join(data_dir, f'{lang}.dev'))
    test_data  = load_data(os.path.join(data_dir, f'{lang}.tst'))

    train_loader = DataLoader(
        InflectionDataset(train_data), batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        InflectionDataset(dev_data), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        InflectionDataset(test_data), batch_size=1,
        shuffle=False, collate_fn=collate_fn
    )
    return train_loader, dev_loader, test_loader
