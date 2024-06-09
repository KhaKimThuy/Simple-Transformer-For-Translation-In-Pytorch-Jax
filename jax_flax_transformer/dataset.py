from constants import EOS_TOKEN, UNK_TOKEN, PAD_TOKEN, \
                    BOS_TOKEN, SRC_LANG, TGT_LANG, \
                    DATASET_CARD

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import re
from tqdm import tqdm
from datasets import load_dataset
import os
import numpy as np

def clean(caption):
    # Remove non-alphabetical character
    caption = re.sub(r'[^a-zA-Z]', r' ', caption)
    # Remove one word character
    caption = re.sub(r'\b[a-zA-Z]\b', r' ', caption)
    # Remove multiple spaces
    caption = re.sub(r'\s+', r' ', caption)
    return caption.strip().lower()

class Vocabulary:
  def __init__(self, vocab_path):
      self.vocab_path = vocab_path
      self.encode_vocab = defaultdict() # text -> index
      self.decode_vocab = defaultdict() # index -> text
      self.build_vocab_dict()
      self.size = self.get_vocab_size()

  def build_vocab_dict(self):
      with open(self.vocab_path, "r") as f:
          for i, word in enumerate(f):
            word = word.strip()
            self.encode_vocab[word] = i
            self.decode_vocab[i] = word

  def encode(self, sentence):
      """
        text -> indices
      """
      word_list = sentence.split()
      indices = [self.encode_vocab[i] for i in word_list]
      return indices

  def decode(self, indices):
      """
        indices -> text
      """
      word_list = [self.decode_vocab[i] for i in indices]
      return word_list

  def get_vocab_size(self):
      return len(self.encode_vocab)

class TranslateDataset(Dataset):
    def __init__(self, dataset, src_vocab, tgt_vocab, transform=clean):
        self.dataset = dataset['translation']
        self.transform = transform
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __getitem__(self, index):
        data = self.dataset[index]
        src, trg = data[SRC_LANG], data[TGT_LANG]
        src, trg = self.transform(src), self.transform(trg)

        src = BOS_TOKEN + " " + src + " " + EOS_TOKEN
        trg = BOS_TOKEN + " " + trg + " " + EOS_TOKEN

        enc_src, enc_trg = self.src_vocab.encode(src), self.tgt_vocab.encode(trg)
        return {'src': enc_src, 'trg': enc_trg}

    def __len__(self):
        return len(self.dataset)


def pad_sequence(sequence, max_length, padding_value=1):
    return sequence + [padding_value] * (max_length - len(sequence))

def collate_fn(batch, padding_value=1):
    source_batch = [item['src'] for item in batch]
    target_batch = [item['trg'] for item in batch]
    
    source_lengths = [len(seq) for seq in source_batch]
    target_lengths = [len(seq) for seq in target_batch]
    
    max_source_length = max(source_lengths)
    max_target_length = max(target_lengths)
    
    padded_sources = [pad_sequence(seq, max_source_length, padding_value) for seq in source_batch]
    padded_targets = [pad_sequence(seq, max_target_length, padding_value) for seq in target_batch]
    
    return np.array(padded_sources), np.array(padded_targets)


def get_raw_vocab(sentence_list, args, src=True):
    vocab_file_name = args["src_vocab_path"] if src else args["trg_vocab_path"]
    unique_words = set()
    for sentence in tqdm(sentence_list):
        words = sentence.strip().split()
        for i, w in enumerate(words):
            words[i] = w
        unique_words.update(words)

    unique_words = list(unique_words)
    unique_words = sorted(unique_words)
    
    with open(vocab_file_name, mode="w", encoding="utf8") as t_file:
        t_file.write(f"{UNK_TOKEN}\n")
        t_file.write(f"{PAD_TOKEN}\n")
        t_file.write(f"{BOS_TOKEN}\n")
        t_file.write(f"{EOS_TOKEN}\n")
        for s in unique_words:
            t_file.write(f"{s}\n")
        print(f"Saved vocabulary to {vocab_file_name}")

def get_vocab(data, args):
    if not os.path.exists(args["trg_vocab_path"]):
        os.makedirs(f"vocab", exist_ok=True)
        print("Create prepare dataset ...")
        src_sentence_list = []
        trg_sentence_list = []
        for i in data["train"]["translation"]:
            src_sentence_list.append(clean(i[SRC_LANG]))
            trg_sentence_list.append(clean(i[TGT_LANG]))

        get_raw_vocab(src_sentence_list, args, src=True)
        get_raw_vocab(trg_sentence_list, args, src=False)

    src_vocab = Vocabulary(args["src_vocab_path"])
    trg_vocab = Vocabulary(args["trg_vocab_path"])
    return src_vocab, trg_vocab

def get_data_loader(args):
    data = load_dataset(DATASET_CARD, f"{SRC_LANG}-{TGT_LANG}")
    src_vocab, tgt_vocab = get_vocab(data, args)
    
    train_val_data = data["train"].train_test_split(test_size=0.2)
    val_test_data = train_val_data["test"].train_test_split(test_size=0.2)

    train_set = train_val_data["train"]
    val_set = val_test_data["train"]
    test_set = val_test_data["test"]

    train_data = TranslateDataset(train_set, src_vocab, tgt_vocab)
    val_data = TranslateDataset(val_set, src_vocab, tgt_vocab)
    test_data = TranslateDataset(test_set, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_data,
                            batch_size=args["batch_size"],
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=collate_fn)

    val_loader = DataLoader(val_data,
                            batch_size=args["batch_size"],
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_data,
                            batch_size=args["batch_size"],
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=collate_fn)

    return src_vocab, tgt_vocab, train_loader, val_loader, test_loader


if __name__ == "__main__":
    data = load_dataset("{DATASET_CARD}", f"{SRC_LANG}-{TGT_LANG}")
    train_val_data = data["train"].train_test_split(test_size=0.2)
    val_test_data = train_val_data["test"].train_test_split(test_size=0.2)

