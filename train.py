import os
import numpy as np
import torch
import torch.nn as nn
import datetime
import sentencepiece as spm


from tqdm import tqdm
from config.config import parse_option
from model import *
from utils.utils import get_data_loader, train_sentencepiece
from dataset.dataset import pad_or_truncate, NMTDataset
from dataset.prepare import DataPreparing
from datasets import load_metric
import transformers
class Trainer():
    def __init__(self, cfg, is_train=True, load_ckpt=False):
        self.cfg = cfg
        print("Loading Transformer model & Adam optimizer...")
        self.model = Transformer(self.cfg).to(self.cfg.device)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

        self.best_loss = 100.0

        if load_ckpt:
            print("Loading checkpoint...")
            checkpoint = torch.load(f'{self.cfg.ckpt_path}/{self.cfg.ckpt_name}', map_location=self.cfg.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
        
        self.prepare_tokenizer()

        if is_train:
            print("Loading loss function...")
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


    def prepare_tokenizer(self):
        if not os.path.isdir(self.cfg.sp_dir):
            print('Training sentencepiece tokenizer...')
            train_sentencepiece(self.cfg, is_src=True)
            train_sentencepiece(self.cfg, is_src=False)

        else:
            print('Tokenization already...')
    
    def train(self):
        print("Training...")

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
            bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc='VALIDATIION')
            for batch_idx, batch in bar:
                src_input, tgt_input, tgt_output = batch['input_ids'], batch['input_tgt_data'], batch['output_tgt_data']
                src_input = src_input.to(self.cfg.device)
                tgt_input = tgt_input.to(self.cfg.device)
                tgt_output = tgt_output.to(self.cfg.device)

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
                if self.cfg.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.cfg.device == 'mps':
                    from torch import mps
                    mps.empty_cache()
 
        end_time = datetime.datetime.now()
        validation_time = end_time - start_time
        
        mean_valid_loss = np.mean(valid_losses)
        
        return mean_valid_loss, f"{validation_time} secs"

    def inference(self, input_sentence):
        self.model.eval()

        print("Preprocessing input sentence...")
        tokenized = self.sp_src.EncodeAsIds(input_sentence)
        src_data = torch.LongTensor(
            pad_or_truncate([self.cfg.sos_id] + tokenized + [self.cfg.eos_id], self.cfg.seq_len, self.cfg.pad_id)
        ).unsqueeze(0).to(self.cfg.device)

        e_mask = (src_data != self.cfg.pad_id).unsqueeze(1).to(self.cfg.device) # (1, 1, L)
        start_time = datetime.datetime.now()

        print("Encoding input sentence...")
        src_data = self.model.src_embedding(src_data)
        src_data = self.model.positional_encoder(src_data)
        e_output = self.model.encoder(src_data, e_mask) # (1, L, d_model)

        result = self.greedy_search(e_output, e_mask)
        end_time = datetime.datetime.now()

        total_inference_time = end_time - start_time
        print(f"Input: {input_sentence}")
        print(f"Result: {result}")
        print(f"Inference finished! || Total inference time: {total_inference_time}secs")
        return result     

    def greedy_search(self, e_output, e_mask):
        last_words = torch.LongTensor([self.cfg.pad_id] * self.cfg.seq_len).to(self.cfg.device) # (L)
        last_words[0] = self.cfg.sos_id # (L)
        cur_len = 1

        for i in range(self.cfg.seq_len):
            d_mask = (last_words.unsqueeze(0) != self.cfg.pad_id).unsqueeze(1).to(self.cfg.device) # (1, 1, L)
            nopeak_mask = torch.ones([1, self.cfg.seq_len, self.cfg.seq_len], dtype=torch.bool).to(self.cfg.device)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

            tgt_embedded = self.model.tgt_embedding(last_words.unsqueeze(0))
            tgt_positional_encoded = self.model.positional_encoder(tgt_embedded)
            decoder_output = self.model.decoder(
                tgt_positional_encoded,
                e_output,
                e_mask,
                d_mask
            ) # (1, L, d_model)

            output = self.model.softmax(
                self.model.output_linear(decoder_output)
            ) # (1, L, trg_vocab_size)

            output = torch.argmax(output, dim=-1) # (1, L)
            last_word_id = output[0][i].item()
            
            if i < self.cfg.seq_len - 1:
                last_words[i+1] = last_word_id
                cur_len += 1
            
            if last_word_id == self.cfg.eos_id:
                break

        if last_words[-1].item() == self.cfg.pad_id:
            decoded_output = last_words[1:cur_len].tolist()
        else:
            decoded_output = last_words[1:].tolist()
        decoded_output = self.sp_tgt.decode_ids(decoded_output)
        
        return decoded_output

    def create_mask(self, src_input, tgt_input):
        e_mask = (src_input != self.cfg.pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (tgt_input != self.cfg.pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, self.cfg.seq_len, self.cfg.seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(self.cfg.device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask  
     
metric = load_metric('sacrebleu')
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = cfg.tokenizer.batch_decode(preds, skip_special_tokens=True)

    decoded_labels = cfg.tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != cfg.tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    
    return result

if __name__ == '__main__':
    cfg = parse_option()

    if not cfg.is_prepare:
        data_pre = DataPreparing(cfg.data_dir, cfg.src_lang, cfg.tgt_lang)
        data_pre.download_dataset()

    if cfg.model_name == 'Transformer':
        trainer = Trainer(cfg, is_train=True, load_ckpt=cfg.load_ckpt)
        trainer.train()
    elif cfg.model_name == 'mBART50':
        training_args = transformers.Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            save_strategy='steps',
            save_steps=cfg.eval_steps,
            eval_steps=cfg.eval_steps,
            output_dir=cfg.ckpt_path,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=0.005,
            num_train_epochs=cfg.epochs,
        )        
        cfg.tokenizer = transformers.MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', src_lang="vi_VN",tgt_lang = "en_XX")
        model = transformers.MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')        
        train_dataset = NMTDataset(cfg, data_type="train")
        valid_dataset = NMTDataset(cfg, data_type="validation")
        test_dataset = NMTDataset(cfg, data_type="test")

        # data_collator = transformers.DataCollatorForSeq2Seq(
        #     cfg.tokenizer, 
        #     model=model
        # )
        
        trainer = transformers.Seq2SeqTrainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            # data_collator=data_collator,
            tokenizer=cfg.tokenizer,
            compute_metrics=compute_metrics
        )       
        print("BEGIN TRAIN")
        trainer.train() 