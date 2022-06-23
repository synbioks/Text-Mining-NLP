ROOT="/sbksvol/nikhil/sbks-ucsd-test/top-model/huggingface-finetune/"
NAME="main"

WANDB_API_KEY = "90852721fdf4fb388c7f75ad45a5a0629bfc4bbf"
export WANDB_API_KEY

nvidia-smi

/root/anaconda3/bin/python3.7 -m pip install --upgrade pip
/root/anaconda3/bin/python3.7 -m pip install transformers==4.6.0
/root/anaconda3/bin/python3.7 -m pip install pytorch-crf
/root/anaconda3/bin/python3.7 -m pip install "ray[tune]"
/root/anaconda3/bin/python3.7 -m pip install wandb

gene=(BC2GM  bioinfer  cellfinder  deca  fsu  iepa  jnlpba  loctext  miRNA  osiris  variome gpro)
cellline=(cellfinder  cll  gellus  jnlpba)
species=(linneaus  loctext  miRNA  s800  variome)
chemicals=(cdr  cemp  chebi  chemdner  scai_chemicals)

declare -A data=(["Gene"]=${gene[*]} ["Cellline"]=${cellline[*]} ["Species"]=${species[*]} ["Chemicals"]=${chemicals[*]})

for var in $(seq 1 $5)
do
    for var2 in ${data[$2]}
    do
        nvidia-smi
        nvidia-smi --gpu-reset -i 0
        st=`date +%s`
        echo "**********************************************************************************************************************"
        echo "#################################### NOW STARTING $2 $var $var2 ####################################"
        /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 42 --entity_type $2 --dataset $var2 --data "/sbksvol/nikhil/" --exp_name "$2-$var2-$var-$1-$st" --exp_config $4 --root "/sbksvol/nikhil/"
    done
done

