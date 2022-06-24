ROOT="/sbksvol/nikhil/sbks-ucsd-test/top-model/huggingface-finetune/biobert/biobert-pytorch/named-entity-recognition/"
NAME="run_ner"

WANDB_API_KEY = "90852721fdf4fb388c7f75ad45a5a0629bfc4bbf"
export WANDB_API_KEY

LABEL_DIR="/sbksvol/nikhil/sbks-ucsd-test/top-model/huggingface-finetune/biobert/biobert-pytorch/datasets/NER/BC2GM"

/root/anaconda3/bin/python3.7 -m pip install --upgrade pip
/root/anaconda3/bin/python3.7 -m pip install torch==1.10 torchvision torchaudio
/root/anaconda3/bin/python3.7 -m pip install transformers==4.6.0
/root/anaconda3/bin/python3.7 -m pip install pytorch-crf
/root/anaconda3/bin/python3.7 -m pip install "ray[tune]"
/root/anaconda3/bin/python3.7 -m pip install wandb

gene=(gpro)
# gene=(BC2GM  bioinfer  cellfinder  deca  fsu  iepa  jnlpba  loctext  miRNA  osiris  variome gpro)
cellline=(cellfinder  cll  gellus  jnlpba)
species=(linneaus  loctext  miRNA  s800  variome)
chemicals=(cdr  cemp  chebi  chemdner  scai_chemicals)

declare -A data=(["Gene"]=${gene[*]} ["Cellline"]=${cellline[*]} ["Species"]=${species[*]} ["Chemicals"]=${chemicals[*]})

for var in $(seq 1 $1)
do
    for var2 in ${data[$2]}
    do
        st=`date +%s`
        echo "**********************************************************************************************************************"
        echo "#################################### NOW STARTING $2 $var $var2 ####################################"
        DATA_DIR=/sbksvol/nikhil/NER_data/
        ENTITY=$2/$var2
        /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" \
        --data_dir ${DATA_DIR}/${ENTITY} \
        --labels ${LABEL_DIR}/labels.txt \
        --model_name_or_path /sbksvol/nikhil/model/biobert_v1.0_pubmed_pmc/ \
        --output_dir /sbksvol/nikhil/output/${ENTITY}/$var \
        --max_seq_length 128 \
        --num_train_epochs 10 \
        --per_device_train_batch_size 32 \
        --save_steps 1000 \
        --do_train \
        --do_eval \
        --do_predict \
        --overwrite_output_dir --overwrite_cache
    done
done
