# Author : Samantha Mahendran for RelEx
#program to convert brat files to csv files
import os
import fnmatch
import pandas as pd

# path to folder of brat files
input_folder= '../data/drugprot/sample/dev/'
# path to output folder
output_folder = '../data/drugprot/predictions/csv/'
output_file_name = 'pred_relations.tsv'

# dataframe and colums for the relation predicitions
column_names = ["file", "label", "e1", "e2"]
relation_preds = pd.DataFrame(columns=column_names)

# read brat files (both entities and relations) into dictionary
for f in os.listdir(input_folder):
    if fnmatch.fnmatch(f, '*.ann'):
        print(f)
        df = pd.DataFrame(columns=column_names)
        # dict to store entities and relations
        annotations = {'entities': {}, 'relations': {}}
        with open(input_folder + str(f), 'r') as file:
            annotation_text = file.read()

        for line in annotation_text.split("\n"):
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if "\t" not in line:
                raise InvalidAnnotationError("Line chunks in ANN files are separated by tabs, see BRAT guidelines. %s"
                                             % line)
            line = line.split("\t")
            if 'T' == line[0][0]:
                tags = line[1].split(" ")
                entity_name = tags[0]
                entity_start = int(tags[1])
                entity_end = int(tags[-1])
                annotations['entities'][line[0]] = (entity_name, entity_start, entity_end, line[-1])

            if 'R' == line[0][0]:  # TODO TEST THIS
                tags = line[1].split(" ")
                assert len(tags) == 3, "Incorrectly formatted relation line in ANN file"
                relation_name = tags[0]
                relation_start = tags[1].split(':')[1]
                relation_end = tags[2].split(':')[1]
                annotations['relations'][line[0]] = (relation_name, relation_start, relation_end)

        # write relations into a pandas dataframe
        for key in annotations['relations']:
            for label_rel, entity1, entity2 in [annotations['relations'][key]]:
                key  = str(key).replace('R', '')
                file = f.replace('.ann', '')
                e1 = 'Arg1:'+ entity1
                e2 = 'Arg2:'+ entity2
                df.loc[key] = [file, label_rel, e1, e2]
        relation_preds = relation_preds.append(df, ignore_index=True)

# convert dataframe into csv file
pmid = relation_preds.file.unique()
f = open(output_folder + str('pmids.txt'), "a")
for i in pmid:
    f.write(i+ '\n')
f.close()
relation_preds.to_csv(output_folder + output_file_name, sep ='\t', index=False, header=False)