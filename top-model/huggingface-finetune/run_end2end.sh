ROOT="/sbksvol/xiang/sbks-ucsd/top-model/huggingface-finetune/"
NAME="run_end2end"
DIR="model_output_end"

pip install pytorch-crf

rm -r "${ROOT}${DIR}"
mkdir "${ROOT}${DIR}"


for var in 0 1 2 3 4
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### NOW STARTING miRNA e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset miRNA    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
done


for var in 0 1 2 3 4
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### NOW STARTING osiris e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset osiris    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
done


for var in 0 1 2 3 4
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### NOW STARTING fsu e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset fsu    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
done

# for var in 0 1 2 3 4
# do
#     echo "**********************************************************************************************************************"
#     echo $var
#     echo "#################################### NOW STARTING cemp e2e ####################################"
#     /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset cemp    
#     rm -r "${ROOT}${DIR}"
#     mkdir "${ROOT}${DIR}"
# done
