export RE_DIR=/sbksvol/jiawei/biobert/datasets/CHEMPROT
export TASK_NAME=chemprot
export OUTPUT_DIR=/sbksvol/jiawei/biobert/outputs
export BIOBERT_DIR=/sbksvol/jiawei/biobert/weights/

python run_re.py --task_name=$TASK_NAME --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/biobert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=2e-5 --num_train_epochs=5.0 --do_lower_case=false --data_dir=$RE_DIR --output_dir=$OUTPUT_DIR
