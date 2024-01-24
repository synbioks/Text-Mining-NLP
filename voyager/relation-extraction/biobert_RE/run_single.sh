export PYTHONPATH=/home/jbentley2/installs/v1.9/tf/local/lib/python3.10/dist-packages:/usr/lib/habanalabs:$PYTHONPATH
export TRANSFORMERS_CACHE=/home/jbentley2/relation-extraction/cache

python3 -m models.train >& single_train_2.log

# tail train.log