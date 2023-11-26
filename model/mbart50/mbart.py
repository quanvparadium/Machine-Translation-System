import numpy as np
from datasets import load_metric
from transformers import (
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    MBart50TokenizerFast, MBartForConditionalGeneration,
    DataCollatorForSeq2Seq
)
from dataset.dataset import pad_or_truncate, NMTDataset
try:
    from accelerate import Accelerator
    accelerator = Accelerator()
except:
    print("Please install 'accelerator'")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels 

def preprocess_function(cfg, examples, max_input_length=128, max_target_length=128):
    inputs = [ex[cfg.src_lang] for ex in examples["translation"]]
    targets = [ex[cfg.tgt_lang] for ex in examples["translation"]]
    model_inputs = cfg.tokenizer(inputs, max_new_tokens=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    # no need this line
    # with tokenizer.as_target_tokenizer():
    labels = cfg.tokenizer(targets, max_new_tokens=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class mBART50():
    def __init__(self, cfg, is_train=True, load_ckpt=False):
        print("LOADING METRIC...")
        self.cfg = cfg
        self.model = None
        self.metric = load_metric('sacrebleu')
        self.training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            save_strategy='steps',
            save_steps=self.cfg.eval_steps,
            eval_steps=self.cfg.eval_steps,
            output_dir=self.cfg.ckpt_path,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate,
            weight_decay=0.005,
            num_train_epochs=self.cfg.epochs,
        )     
        self.cfg.tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', src_lang="vi_VN",tgt_lang = "en_XX")

        # self.prepare_dataset()
    
        if load_ckpt:
            print("LOADING CHECKPOINT...")
            self.model = MBartForConditionalGeneration.from_pretrained(self.cfg.ckpt_path)
        else:
            print("DEVICE: ", self.cfg.device)
            self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
        
        self.data_collator = DataCollatorForSeq2Seq(
            self.cfg.tokenizer, 
            model=self.model
        )
        assert self.model is not None
        train_dataset, valid_dataset, _ = self.prepare_dataset()
        
        self.trainer = Seq2SeqTrainer(
            self.model,
            self.training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self.data_collator,
            tokenizer=self.cfg.tokenizer,
            compute_metrics=self.compute_metrics
        )        


    def prepare_dataset(self):
        train_dataset = NMTDataset(self.cfg, data_type="train")
        valid_dataset = NMTDataset(self.cfg, data_type="validation")
        test_dataset = NMTDataset(self.cfg, data_type="test")

        return train_dataset, valid_dataset, test_dataset

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.cfg.tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = self.cfg.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.cfg.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        
        return result
    
    def train(self):
   

        if self.cfg.accelerator:
            print("ACCELERATING...")
            train_dataset, valid_dataset, self.trainer = accelerator.prepare(
                train_dataset, valid_dataset, self.trainer
            )
        print("MODEL TRAINING...")            
        self.trainer.train()

    def evaluate(self):
        print("MODEL EVALUATING...")
        self.trainer.evaluate()

    def evaluate(self):
        print("MODEL EVALUATING...")
        self.trainer.evaluate()    

    def inference(self, input_text):
        self.cfg.tokenizer.src_lang = "vi_VN"
        encoding = self.cfg.tokenizer(input_text, return_tensors="pt")   
        outputs = self.model.generate(**encoding)
        predictions = self.cfg.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("RESULT: ", predictions)