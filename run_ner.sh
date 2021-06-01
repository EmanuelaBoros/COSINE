task=ner
gpu=1
method=noise
max_seq_len=128
batch_size=16
lang=en
echo ${method}

rm -rf data/ner/cached_*

python3 main.py \
	--do_train \
	--do_eval \
	--task=${task} \
	--data_dir="data/${task}/${lang}" \
	--task_type='tc' \
	--rule=0 \
	--logging_steps=100 \
	--self_train_logging_steps=100 \
	--gpu="${gpu}" \
	--num_train_epochs=5 \
	--weight_decay=1e-4 \
	--method=${method} \
	--batch_size=${batch_size} \
	--max_seq_len=${max_seq_len} \
	--auto_load=1 \
	--learning_rate=5e-5 \
        --model_type roberta
	# --self_training_update_period=250 \
	# --self_training_max_step=2500 \
	# --self_training_power=2 \
	# --self_training_confreg=0.1 \
	# --self_training_contrastive_weight=1 \
	# --distmetric='cos' \
	# --max_steps=150 \

