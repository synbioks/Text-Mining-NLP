ROOT="/sbksvol/xiang/sbks-ucsd/top-model/huggingface-finetune/"
NAME="run_baseline"
DIR="model_output_baseline"

rm -r "${ROOT}${DIR}"
mkdir "${ROOT}${DIR}"


for var in 0 1
do
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Cellline cellfinder baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Cellline --dataset cellfinder   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Cellline cll baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Cellline --dataset cll   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Cellline gellus baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Cellline --dataset gellus   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Cellline jnlpba baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Cellline --dataset jnlpba   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    










    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals biosemantics baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset biosemantics    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals cdr baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset cdr    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals cemp baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset cemp    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals chebi baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset chebi    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals chemdner baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset chemdner   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Chemicals scai_chemicals baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Chemicals --dataset scai_chemicals    
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    
    
    
    
    
    
    
    
    
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene BC2GM baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset BC2GM   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene bioinfer baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset bioinfer   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene cellfinder baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset cellfinder   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene deca baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset deca   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
     
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene fsu baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset fsu   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene iepa baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset iepa   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene jnlpba baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset jnlpba   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene loctext baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset loctext   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene miRNA baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset miRNA   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene osiris baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset osiris   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Gene variome baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Gene --dataset variome   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    
    
    
    
    
    
    
    
    
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species cellfinder baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset cellfinder   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species linneaus baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset linneaus   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species loctext baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset loctext   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species miRNA baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset miRNA   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species s800 baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset s800   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
    
    echo "**********************************************************************************************************************"
    echo $var
    echo "#################################### Species variome baseline ####################################"
    /root/anaconda3/bin/python3.7 "${ROOT}${NAME}.py" --seed_value 92 --set_seed NO --entity_type Species --dataset variome   
    rm -r "${ROOT}${DIR}"
    mkdir "${ROOT}${DIR}"
        
done

