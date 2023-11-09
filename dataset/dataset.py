import torch.utils.data as data
import sentencepiece as spm
from tqdm import tqdm
import torch
import numpy as np
def pad_or_truncate(tokenized_sequence, seq_len, pad_id):
    if len(tokenized_sequence) < seq_len:
        left = seq_len - len(tokenized_sequence)
        padding = [pad_id] * left
        tokenized_sequence += padding
    else:
        tokenized_sequence = tokenized_sequence[:seq_len]
    return tokenized_sequence
class NMTDataset(data.Dataset):
    def __init__(self, cfg, data_type='train'):
        super().__init__()
        self.cfg = cfg

        self.sp_src, self.sp_tgt = self.load_sp_tokenizer()
        self.src_texts, self.tgt_texts = self.read_data(data_type)

        if self.cfg.model_name == "Transformer":
            src_tokenized_sequences = self.texts_to_sequences(self.src_texts, is_src=True)
            tgt_input_tokenized_sequences, tgt_output_tokenized_sequences = self.texts_to_sequences(self.tgt_texts, is_src=False)
            self.input_tgt_data = torch.LongTensor(tgt_input_tokenized_sequences)
            self.output_tgt_data = torch.LongTensor(tgt_output_tokenized_sequences)
        else:
            src_tokenized_sequences = self.texts_to_sequences(self.src_texts)
            tgt_tokenized_sequences = self.texts_to_sequences(self.tgt_texts)
            self.labels = torch.LongTensor(tgt_tokenized_sequences)

        self.src_data = torch.LongTensor(src_tokenized_sequences)


    def read_data(self, data_type):
        print(f"===> Load data from: {self.cfg.data_dir}/{data_type}.{self.cfg.src_lang}")
        with open(f'{self.cfg.data_dir}/{data_type}.{self.cfg.src_lang}', 'r', encoding='utf-8') as f:
            src_texts = f.readlines()
        
        print(f"===> Load data from: {self.cfg.data_dir}/{data_type}.{self.cfg.tgt_lang}")
        with open(f"{self.cfg.data_dir}/{data_type}.{self.cfg.tgt_lang}", 'r', encoding='utf-8') as f:
            trg_texts = f.readlines()
        
        return src_texts, trg_texts        
    
    def load_sp_tokenizer(self):
        sp_src = spm.SentencePieceProcessor()
        sp_src.Load(f"{self.cfg.sp_dir}/{self.cfg.src_model_prefix}.model")

        sp_tgt = spm.SentencePieceProcessor()
        sp_tgt.Load(f"{self.cfg.sp_dir}/{self.cfg.tgt_model_prefix}.model")

        return sp_src, sp_tgt

    def texts_to_sequences(self, texts, is_src=True):
        if self.cfg.model_name == "Transformer":
            if is_src:
                src_tokenized_sequences = []
                for text in tqdm(texts):
                    tokenized = self.sp_src.EncodeAsIds(text.strip())
                    src_tokenized_sequences.append(
                        pad_or_truncate([self.cfg.sos_id] + tokenized + [self.cfg.eos_id], self.cfg.seq_len, self.cfg.pad_id)
                    )
                return src_tokenized_sequences
            else:
                tgt_input_tokenized_sequences = []
                tgt_output_tokenized_sequences = []
                for text in tqdm(texts):
                    tokenized = self.sp_tgt.EncodeAsIds(text.strip())
                    tgt_input = [self.cfg.sos_id] + tokenized
                    tgt_output = tokenized + [self.cfg.eos_id]
                    tgt_input_tokenized_sequences.append(pad_or_truncate(tgt_input, self.cfg.seq_len, self.cfg.pad_id))
                    tgt_output_tokenized_sequences.append(pad_or_truncate(tgt_output, self.cfg.seq_len, self.cfg.pad_id))
                
                return tgt_input_tokenized_sequences, tgt_output_tokenized_sequences
        elif self.cfg.model_name == "mBART50":
            data_inputs = self.cfg.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.cfg.seq_len,
                return_tensors='pt'
            )
            return data_inputs.input_ids            
    def __getitem__(self, idx):
        if self.cfg.model_name == "Transformer":
            return {
                "input_ids": self.src_data[idx],
                "input_tgt_data": self.input_tgt_data[idx],
                "output_tgt_data": self.output_tgt_data[idx]
            }
        elif self.cfg.model_name == "mBART50":
            return {
                "input_ids": self.src_data[idx],
                "labels": self.labels[idx]
            }
        else:
            assert True == False

        # return self.src_data[idx], self.input_tgt_data[idx], self.output_tgt_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]