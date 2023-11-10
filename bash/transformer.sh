python train.py \
    --model_name 'Transformer' --model_type 'unigram' --device 'mps'\
    --num_heads 8 --num_layers 6 --d_model 512 --d_ff 2048 --drop_out 0.1 \\
    --batch_size 16 --epochs 1 \
    --src_lang 'vi' --tgt_lang 'en' --seq_len 150 \
    --load_ckpt false \
    --ckpt_path 'pretrained' --ckpt_name 'transformer_ep1.pt'\