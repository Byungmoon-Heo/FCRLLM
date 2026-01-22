gpu_id=0
dataset="fashion"

model_name="fcrllm_sasrec"
# model_name="fcrllm_gru4rec"

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
        --patience 20 \
        --ts_user 3 \
        --ts_item 4 \
        --freeze \
        --log \
        --gamma_fc 0.5 \
        --beta_hopfield 0.7 \
        --alpha_hopfield 0.7 \
        --gamma_update 0.3 \
        --do_test
                