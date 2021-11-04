To import local packages: make sure to add the path of this folder to PYTHONPATH

To set up vscode debugging with argument, create run configurations


command copy-paste board:

process drugprot dev
```
python ../../../biobert_RE_v2/dataset_processing/dataset_to_json.py ./drugprot_development_entities.tsv ./drugprot_development_relations.tsv ./drugprot_development_abstracts.tsv -o ./drugprot_dev.json -v
```

process drugprot train
```
python ../../../biobert_RE_v2/dataset_processing/dataset_to_json.py ./drugprot_training_entities.tsv ./drugprot_training_relations.tsv ./drugprot_training_abstracts.tsv -o ./drugprot_train.json -v
```

process chemprot dev
```
python ../../../biobert_RE_v2/dataset_processing/dataset_to_json.py ./chemprot_development_entities.tsv ./chemprot_development_relations.tsv ./chemprot_development_abstracts.tsv -o ./chemprot_dev.json -v
```

process chemprot train
```
python ../../../biobert_RE_v2/dataset_processing/dataset_to_json.py ./chemprot_training_entities.tsv ./chemprot_training_relations.tsv ./chemprot_training_abstracts.tsv -o ./chemprot_train.json -v
```

Some notes:

1. article ids are unique even across datasets
2. drugprot train and dev do not share any common article ids (we assume this is the case for other datasets as well)
3. the abstract in .json datasets are actually abstract + title (the spans start from title)
4. entities and relations in abstract sentences are sorted by start position
5. chemprot and drugprot has the same exact abstracts, and entities