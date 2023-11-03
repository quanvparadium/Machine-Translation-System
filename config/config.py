import argparse
import math
import os
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency') 
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--src_lang', type=str, default='vi')
    parser.add_argument('--tgt_lang', type=str, default='en')
    parser.add_argument('--dataset_name', type=str, default='mt_eng_vietnamese')

    parser.add_argument('--model_name', type=str, default='Transformer', choices=['Transformer', 'mBART50', 'BERT-GPT'])
    parser.add_argument('--model_type', type=str, default='unigram', choices=['BPE', 'unigram', 'char', 'word'])
    parser.add_argument('--pad_id', type=int, default=0)
    parser.add_argument('--sos_id', type=int, default=1)
    parser.add_argument('--eos_id', type=int, default=2)
    parser.add_argument('--unk_id', type=int, default=3)

    parser.add_argument('--seq_len', type=int, default=150, help="Max sequence length")
    parser.add_argument('--sp_dir', type=str, default='', help='Sentencepiece model')
    parser.add_argument('--sp_vocab_size', type=int, default=4000)
    parser.add_argument('--character_coverage', type=float, default=1.0)

    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--drop_out', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=50)

    parser.add_argument('--load_ckpt', type=bool, default=False, help='path to pre-trained model')
    parser.add_argument('--ckpt_path', type=str, default='', help='path to pre-trained model')
    parser.add_argument('--ckpt_name', type=str, default='')

    parser.add_argument('--is_prepare', type=bool, default=False, help='Download dataset if not exist dataset')
    opt = parser.parse_args()

    if not os.path.isdir(opt.data_dir):
        os.mkdir(opt.data_dir)
        
    if opt.device is None:
        import torch
        opt.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if opt.data_dir:
        opt.sp_dir = opt.data_dir + '/sp'
        opt.src_model_prefix = 'sp_' + opt.src_lang
        opt.tgt_model_prefix = 'sp_' + opt.tgt_lang

    # opt.train_batch_size = opt.batch_size
    # opt.eval_batch_size = opt.batch_size

    return opt