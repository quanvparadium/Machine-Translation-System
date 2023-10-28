import os

from dataset.dataset import NMTDataset, pad_or_truncate
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

def get_data_loader(cfg, data_type='train'):
    dataset = NMTDataset(cfg, data_type)

    if data_type == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)

    return dataset, dataloader

def train_sentencepiece(cfg, is_src=True):
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"
    
    if is_src:
        train_file = f"{cfg.data_dir}/train.{cfg.src_lang}"
        model_prefix = f"{cfg.sp_dir}/{cfg.src_model_prefix}"
    else:
        train_file = f"{cfg.data_dir}/train.{cfg.tgt_lang}"
        model_prefix = f"{cfg.sp_dir}/{cfg.tgt_model_prefix}"

    print(f"===> Processing file: {train_file}")
    if not os.path.isdir(cfg.sp_dir):
        os.mkdir(cfg.sp_dir)

    sp_cfg = template.format(
        train_file,
        cfg.pad_id,
        cfg.sos_id,
        cfg.eos_id,
        cfg.unk_id,
        model_prefix,
        cfg.sp_vocab_size,
        cfg.character_coverage,
        cfg.model_type)
    
    spm.SentencePieceTrainer.Train(sp_cfg)