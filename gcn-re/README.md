# RelEx-GCN

This code is cloned from this [repository](https://github.com/NLPatVCU/RelEx-GCN/tree/main/GCN-BERT) and is an implementation of this [paper](https://dl.acm.org/doi/abs/10.1145/3487553.3524702)

* `data`: contains the merged Chemprot and Drugprot datasets in BratEval format. Refer to the [relation-extraction](https://gitlab.com/mhn-ucsd/sbks-ucsd/-/blob/master/relation-extraction/README.md) folder for data processing details
* `GCN-BERT`: contains the data processing, segmentaion and GCN-modeling files
* `docker`: contains DockerFile and yaml files

run.py is the entrypoint to this code. The config file for our dataset is drugprot.ini