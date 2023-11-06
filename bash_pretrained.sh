python train_new.py \
    --model_name 'mBART50' \
    --src_lang 'vi' --tgt_lang 'en' --seq_len 100 \
    --batch_size 16 --epochs 2 --eval_steps 500\
    --learning_rate 0.00005 --ckpt_path 'pretrained'\
    --mode 'evaluate'\