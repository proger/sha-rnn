# prepare the dataset for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
# see also prepare.py in karpathy/nanoGPT

import mmap
import sentencepiece as spm
from datasets import load_dataset, Value, Features
import random
from tqdm import tqdm
import ftfy
import numpy as np
import os

num_proc = 24
dataset = load_dataset("json",
                       # original files have too many different columns:
                       #data_files={"train": ["ubertext.fiction.filter_rus_gcld+short.orig.jsonl", "ubertext.news.filter_rus_gcld+short.orig.jsonl", "ubertext.wikipedia.filter_rus_gcld+short.orig.jsonl"]},

                       # instead do this:
                       # cat *jsonl | jq -rc '{text,title,date_of_publish,tags}' > ubertext.jsonl
                       data_files={"train": ["ubertext.jsonl"]}, 
                       features=Features(
                                {'text': Value(dtype='string', id=None),
                                 'title': Value(dtype='string', id=None),
                                 'date_of_publish': Value(dtype='string', id=None),
                                 'tags': [Value(dtype='string', id=None)]}))

# create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val


def process(example):
    text, title, date_of_publish, tags = example['text'], example.get('title'), example.get('date_of_publish'), example.get('tags')
    meta = []
    if title:
        title = ftfy.fix_text(title)
        meta.append(f'тема: {title}')
    if tags:
        tags = ', '.join(tags)
        tags = ftfy.fix_text(tags)
        meta.append(f'мітки: {tags}')
    if date_of_publish:
        year = date_of_publish.split('-')[0]
        meta.append(f'рік: {year}')
    random.shuffle(meta)
    pre = b'\n'.join(m.encode('utf-8') for m in meta)
    random.shuffle(meta)
    post = b'\n'.join(m.encode('utf-8') for m in meta)

    # ascii(7):
    #  Oct   Dec   Hex   Char                        Oct   Dec   Hex   Char
    #  ────────────────────────────────────────────────────────────────────────
    #  000   0     00    NUL '\0' (null character)   100   64    40    @
    #  001   1     01    SOH (start of heading)      101   65    41    A
    #  002   2     02    STX (start of text)         102   66    42    B
    #  003   3     03    ETX (end of text)           103   67    43    C
    #  004   4     04    EOT (end of transmission)   104   68    44    D
    #  005   5     05    ENQ (enquiry)               105   69    45    E
    #  006   6     06    ACK (acknowledge)           106   70    46    F

    text = ftfy.fix_text(text)
    binary = b'\x00' + b'\x01' + pre + b'\x02' + text.encode('utf-8') + b'\x03' + post
    out = {'binary': binary, 'len': len(binary)}
    return out

# format the data strea,
processed = split_dataset.map(
    process,
    desc="streaming",
    remove_columns=['text', 'title', 'date_of_publish', 'tags'],
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in processed.items():
    arr_len = np.sum(dset['len'])
    # preallocate space in a temporary file to store the concatenated ids
    filename = f'{split}.txt.raw'
    with open(filename, 'w+b') as f:
        os.truncate(f.fileno(), arr_len)

        with mmap.mmap(f.fileno(), arr_len) as mm:
            total_batches = min(10240, len(dset)//10)
            idx = 0
            
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                 # Batch together samples for faster write
                 batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                 arr_batch = b''.join(batch['binary'])
                 # Write into mmap
                 mm[idx : idx + len(arr_batch)] = arr_batch
                 idx += len(arr_batch)
