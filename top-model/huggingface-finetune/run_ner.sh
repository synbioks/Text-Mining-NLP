ROOT="/sbksvol/gaurav/NER_src/"
NAME="run_ner"

pip install pytorch-crf

# /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 42 --entity_type Cellline --dataset cll --model dense_layer_softmax
/root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 42 --entity_type Cellline --dataset cellfinder --model dense_layer_crf
# /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 42 --entity_type Cellline --dataset cll --model bilstm_crf