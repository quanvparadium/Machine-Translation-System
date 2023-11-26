import os 
import numpy as np
import torch
import torch.nn as nn
import datetime
from dataset.prepare import DataPreparing
from config.config import parse_option
from model.transformer.trainer import Trainer
from model.mbart50.mbart import mBART50


if __name__ == '__main__':
    cfg = parse_option()
    if not cfg.is_prepare:
        data_pre = DataPreparing(cfg.data_dir, cfg.src_lang, cfg.tgt_lang)
        data_pre.download_dataset()
    if cfg.mode == "train":
        is_train = True
    else:
        is_train = False
    if cfg.model_name == 'Transformer':
        trainer = Trainer(cfg, is_train=is_train, load_ckpt=cfg.load_ckpt)
        if is_train:
            trainer.train()
        else:
            print("INFERENCING TRANSFORMER MODEL...")
            trainer.inference(cfg.input_text)
    elif cfg.model_name == "mBART50":
        trainer = mBART50(cfg, is_train=is_train)
        if cfg.mode == "train":
            trainer.train()
        elif cfg.mode == "evaluate":
            trainer.evaluate()
        else:
            trainer.inference(cfg.input_text)
