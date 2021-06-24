ROOT="/sbksvol/nikhil/sbks-ucsd/top-model/huggingface-finetune/"
NAME="main"

/root/anaconda3/bin/python3.7 -m pip install --upgrade pip
/root/anaconda3/bin/python3.7 -m pip install pytorch-crf


for var in 0 1 2
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### NOW STARTING $3 ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type $2 --dataset $3 --exp_name $1 --exp_config $4 --root "/sbksvol/nikhil/"
done

