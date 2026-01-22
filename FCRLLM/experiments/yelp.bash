gpu_id=0
dataset="yelp"

# model_name="fcrllm_sasrec"
model_name="fcrllm_gru4rec"

python main.py --dataset ${dataset} \
        --model_name ${model_name} \
        --hidden_size 64 \
        --train_batch_size 128 \
        --max_len 200 \
        --gpu_id ${gpu_id} \
        --num_workers 8 \
        --num_train_epochs 200 \
        --seed 42 \
        --check_path "" \
        --patience 10 \
        --ts_user 12 \
        --ts_item 13 \
        --freeze \
        --log \
        --gamma_fc 1.0 \
        --beta_hopfield 0.5 \
        --alpha_hopfield 0.3 \
        --gamma_update 0.3 

