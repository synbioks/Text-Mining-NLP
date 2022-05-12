ROOT="/sbksvol/nikhil/sbks-ucsd-test/top-model/huggingface-finetune/"
NAME="main"

WANDB_API_KEY="90852721fdf4fb388c7f75ad45a5a0629bfc4bbf"
export WANDB_API_KEY


/root/anaconda3/bin/python3.7 -m pip install --upgrade pip
/root/anaconda3/bin/python3.7 -m pip install transformers==4.6.0
/root/anaconda3/bin/python3.7 -m pip install pytorch-crf
/root/anaconda3/bin/python3.7 -m pip install "ray[tune]"
/root/anaconda3/bin/python3.7 -m pip install wandb

for var in $(seq 1 $5)
do
    st=`date +%s`
    echo "**********************************************************************************************************************"
    echo "#################################### NOW STARTING $2 $var $3 ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 42 --entity_type $2 --dataset $3 --data "/sbksvol/nikhil/" --exp_name "$2-$3-$var-$1-$st" --exp_config $4 --root "/sbksvol/nikhil/"
done

