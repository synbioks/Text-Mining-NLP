export PYTHONPATH=/home/jbentley2/installs/v1.9/tf/local/lib/python3.10/dist-packages:/usr/lib/habanalabs:$PYTHONPATH
export PHY_CPU_COUNT=$(lscpu --all --parse=CORE,SOCKET | grep -Ev "^#" | sort -u | wc -l)
export PHY_HPU_COUNT=$(ls /dev/hl? | wc -l)
export MPI_PE=$(($PHY_CPU_COUNT/$PHY_HPU_COUNT))

export TRANSFORMERS_CACHE=/home/jbentley2/relation-extraction/cache
export HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7

mpirun --allow-run-as-root -np 8 python3 -m models.train_multicard >& multicard_train.log