ROOT="/sbksvol/xiang/sbks-ucsd/top-model/huggingface-finetune/"
NAME="run_baseline"
DIR="model_output_baseline"

rm -r "${ROOT}${DIR}"
mkdir "${ROOT}${DIR}"



for var in 0 1 2
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Cellline cellfinder e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Cellline --dataset cellfinder   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Cellline cll e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Cellline --dataset cll   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Cellline gellus e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Cellline --dataset gellus   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Cellline jnlpba e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Cellline --dataset jnlpba   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    










    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals biosemantics e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset biosemantics    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals cdr e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset cdr    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals cemp e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset cemp    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals chebi e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset chebi    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals chemdner e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset chemdner   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals scai_chemicals e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset scai_chemicals    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    
    
    
    
    
    
    
    
    
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene BC2GM e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset BC2GM   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene bioinfer e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset bioinfer   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene cellfinder e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset cellfinder   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene deca e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset deca   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
     
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene fsu e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset fsu   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene iepa e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset iepa   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene jnlpba e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset jnlpba   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene loctext e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset loctext   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene miRNA e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset miRNA   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene osiris e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset osiris   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene variome e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset variome   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    
    
    
    
    
    
    
    
    
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species cellfinder e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset cellfinder   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species linneaus e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset linneaus   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species loctext e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset loctext   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species miRNA e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset miRNA   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species s800 e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset s800   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species variome e2e ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset variome   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
        
done

