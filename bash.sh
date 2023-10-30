python train.py \
    --model_name 'Transformer' --model_type 'unigram' --device 'mps'\
    --num_heads 8 --num_layers 6 --d_model 512 --d_ff 2048 --drop_out 0.1 --batch_size 16 \
    --src_lang 'vi' --tgt_lang 'en' --seq_len 150 \