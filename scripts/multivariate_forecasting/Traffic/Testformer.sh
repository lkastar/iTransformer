export CUDA_VISIBLE_DEVICES=0

model_name=Testformer
desc=${1:-Baseline}
notes=${2:-Test}

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des $desc \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --wandb_notes "$notes" \
  --use_wandb

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des $desc \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --wandb_notes "$notes" \
  --use_wandb

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des $desc \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --wandb_notes "$notes" \
  --use_wandb

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des $desc \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001\
  --itr 1 \
  --wandb_notes "$notes" \
  --use_wandb
