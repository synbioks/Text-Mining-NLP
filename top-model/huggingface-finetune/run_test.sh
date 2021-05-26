ROOT="/sbksvol/xiang/sbks-ucsd/top-model/huggingface-finetune/"
NAME="run_test"
DIR="model_output_test"

pip install pytorch-crf

rm -r "${ROOT}${DIR}"
mkdir "${ROOT}${DIR}"

for var in 0 1 2 3 4
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### NOW STARTING miRNA dense_layer_softmax ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset miRNA    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
done

for var in 0 1 2 3 4
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### NOW STARTING osiris dense_layer_softmax ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset osiris    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
done

for var in 0 1 2 3 4
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### NOW STARTING variome dense_layer_softmax ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset variome    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
done
