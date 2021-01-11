#  python run_re.py --task_name=$TASK_NAME --do_train=false --do_eval=false --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --max_seq_length=128 --data_dir=$RE_DIR --output_dir=$OUTPUT_DIR

export RE_DIR=/sbksvol/udayan/biobert/datasets/processed
export TASK_NAME=chemprot
export OUTPUT_DIR=/sbksvol/udayan/output/5
export BIOBERT_DIR=/sbksvol/udayan/pretrained_models/biobert_v1.0_pubmed_pmc


for f in ${RE_DIR}/*; do
	export RE_DIR=${f}
	python run_re.py --task_name=$TASK_NAME --do_train=false --do_eval=false --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --max_seq_length=128 --data_dir=$RE_DIR --output_dir=$OUTPUT_DIR
	cp $OUTPUT_DIR/test_results.tsv $f/test_results.tsv
done
