import os 
import numpy as np
import torch
import torch.nn as nn
import datetime
import sentencepiece as spm
from tqdm import tqdm

from model.transformer import (
    Transformer,
)


from utils.utils import get_data_loader, train_sentencepiece

class Trainer():
    def __init__(self, cfg, is_train=True, load_ckpt=False):
        self.cfg = cfg

        self.model = Transformer(self.cfg).to(self.cfg.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        
        self.best_loss = 100.0

        if load_ckpt:
            print("LOADING CHECKPOINT...")
            checkpoint = torch.load(f'{self.cfg.ckpt_path}/{self.cfg.ckpt_name}', map_location=self.cfg.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)     

        # Get mode Trainer
        if is_train:
            print("LOADING LOSS FUNCTION...")
            self.criterion = nn.NLLLoss()

            print("Loading dataloaders...")
            self.train_dataset, self.train_loader = get_data_loader(self.cfg, 'train')
            self.valid_dataset, self.valid_loader = get_data_loader(self.cfg, 'validation')             
        else:
            if os.path.exists(f'{self.cfg.ckpt_path}/{self.cfg.ckpt_name}'):
                print("Loading sentenpiece tokenizer")
                self.sp_src = spm.SentencePieceProcessor()
                self.sp_tgt = spm.SentencePieceProcessor()
                self.sp_src.Load(f"{self.cfg.sp_dir}/{self.cfg.src_model_prefix}.model")
                self.sp_tgt.Load(f"{self.cfg.sp_dir}/{self.cfg.tgt_model_prefix}.model")
            else:
                print("Checkpoint path not exists...")
        
        self.prepare_tokenizer()


    def prepare_tokenizer(self):
        if not os.path.isdir(self.cfg.sp_dir):
            print('Training sentencepiece tokenizer...')
            train_sentencepiece(self.cfg, is_src=True)
            train_sentencepiece(self.cfg, is_src=False)

        else:
            print('Tokenization already...')

    def create_mask(self, src_input, tgt_input):
        e_mask = (src_input != self.cfg.pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (tgt_input != self.cfg.pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, self.cfg.seq_len, self.cfg.seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(self.cfg.device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask  

    def train(self):
        print("MODEL TRAINING...")
        for epoch in range(1, self.cfg.epochs+1):
            print(f"Epoch: {epoch}")
            self.model.train()
            train_losses = []  
            start_time = datetime.datetime.now()
            bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='TRAINING')
            
            for batch_idx, batch in bar:
                src_input, tgt_input, tgt_output = batch['input_ids'], batch['input_tgt_data'], batch['output_tgt_data']
                src_input = src_input.to(self.cfg.device)
                tgt_input = tgt_input.to(self.cfg.device)
                tgt_output = tgt_output.to(self.cfg.device)        

                e_mask, d_mask = self.create_mask(src_input, tgt_input)

                logits = self.model(src_input, tgt_input, e_mask, d_mask)

                self.optim.zero_grad()

                loss = self.criterion(
                    logits.view(-1, logits.shape[-1]),
                    tgt_output.reshape(-1)
                )
                
                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())
                
                del src_input, tgt_input, tgt_output, e_mask, d_mask, logits
                if self.cfg.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.cfg.device == 'mps':
                    from torch import mps
                    mps.empty_cache()

                bar.set_postfix(TRAIN="Epoch {} - Batch_Loss {:.2f} - Train_Loss {:.2f} - Best_Valid_Loss {:.2f}".format(
                    epoch,
                    loss.item(),
                    np.mean(train_losses),
                    self.best_loss
                    )
                )
            
            end_time = datetime.datetime.now()
            training_time = end_time - start_time   

            mean_train_loss = np.mean(train_losses)
            print(f"Train loss: {mean_train_loss} || Time: {training_time} secs")

            valid_loss, valid_time = self.validation()
            if valid_loss < self.best_loss:            
                if not os.path.exists(self.cfg.ckpt_path):
                    os.mkdir(self.cfg.ckpt_path)
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'loss': self.best_loss                    
                }
                torch.save(state_dict, f"{self.cfg.ckpt_path}/{self.cfg.ckpt_name}")
                print(f"***** Current best checkpoint is saved. *****")
            
            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || One epoch training time: {valid_time}")
        
        print(f"Training finished!")
        
    def validation(self):
        self.model.eval()
        
        valid_losses = []
        start_time = datetime.datetime.now()

        with torch.no_grad():
            bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc='VALIDATION')
            for batch_idx, batch in bar:
                src_input, tgt_input, tgt_output = batch['input_ids'], batch['input_tgt_data'], batch['output_tgt_data']
                src_input, tgt_input, tgt_output = src_input.to(self.cfg.device), tgt_input.to(self.cfg.device), tgt_output.to(self.cfg.device)

                e_mask, d_mask = self.create_mask(src_input, tgt_input)

                logits = self.model(src_input, tgt_input, e_mask, d_mask)

                loss = self.criterion(
                    logits.view(-1, logits.shape[-1]),
                    tgt_output.reshape(-1)
                )

                valid_losses.append(loss.item())

                bar.set_postfix(TRAIN="Batch_Loss {:.2f} - Valid_Loss {:.2f}".format(
                    loss.item(),
                    np.mean(valid_losses)
                    )
                )

                del src_input, tgt_input, tgt_output, e_mask, d_mask, logits
                torch.cuda.empty_cache()

        end_time = datetime.datetime.now()
        validation_time = end_time - start_time
        
        mean_valid_loss = np.mean(valid_losses)
        
        return mean_valid_loss, f"{validation_time} secs"