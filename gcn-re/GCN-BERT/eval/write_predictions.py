# Author : Samantha Mahendran for RelEx

import os
import numpy as np
import pandas as pd


class Predictions:
    def __init__(self, cleanData, graph, pred, initial_predictions, final_predictions, No_Rel=False, dominant_entity='S'):
        '''
        Write predictions back to files
        :param final_predictions: predicted relations
        :param No_Rel: flag whether to write the relations with No-relation label back to files
        '''
        self.dominant_entity = dominant_entity
        self.No_Rel = No_Rel
        print("No rel:", self.No_Rel)

        # path to the folder to save the predictions
        self.initial_predictions = initial_predictions

        # path to the folder to save the re-ordered predictions to the files where the entities are already appended
        self.final_predictions = final_predictions
        self.cleanData = cleanData
        self.graph = graph

        # Delete all files in the folder before the prediction
        ext = ".ann"
        filelist = [f for f in os.listdir(self.initial_predictions) if f.endswith(ext)]
        for f in filelist:
            os.remove(os.path.join(self.initial_predictions, f))

        self.write_relations(pred)
        self.renumber_relations()

    def write_relations(self, pred):
        """
        write the predicted relations into their respective files
        :param label_list:
        :param shuffle_text:
        :param test_index:
        """
        for x in range (len(self.cleanData.test_data)):
            pred_label = self.graph.class_labels[pred[x]]
            track = self.cleanData.test_data[x].split("\t")[2]
            file = track.strip('][').split(', ')[0].strip("'")
            e1 = track.strip('][').split(', ')[1].strip("'")
            e2 = track.strip('][').split(', ')[2].strip("'")

            if len(str(file)) == 1:
                f = "000" + str(file) + ".ann"
            elif len(str(file)) == 2:
                f = "00" + str(file) + ".ann"
            elif len(str(file)) == 3:
                f = "0" + str(file) + ".ann"
            else:
                f = str(file) + ".ann"
                # print(f)
                # key for relations (not for a document but for all files)
            key = "R" + str(x + 1)
            # entity pair in the relations
            e1 = "T" + str(e1)
            e2 = "T" + str(e2)

            f1 = open( self.initial_predictions + str(f), "a")
            if self.No_Rel:
                # open and append relation the respective files in BRAT format
                f1.write(str(key) + '\t' + str(pred_label) + ' ' + 'Arg1:' + str(e1) + ' ' + 'Arg2:' + str(
                    e2) + '\n')
                f1.close()
            else:
                if pred_label != 'No-Relation':
                    print(f)
                    # open and append relation the respective files in BRAT format
                    if self.dominant_entity == 'S':
                        f1.write(str(key) + '\t' + str(pred_label) + ' ' + 'Arg1:' + str(e1) + ' ' + 'Arg2:' + str(e2) + '\n')
                    else:
                        f1.write(str(key) + '\t' + str(pred_label) + ' ' + 'Arg1:' + str(e2) + ' ' + 'Arg2:' + str(e1) + '\n')
                    f1.close()
            # break

    def renumber_relations(self):
        """
        When writing predictions to file the key of the relations are not ordered based on individual files.
        This function renumbers the appended predicted relations in each file

        :param initial_predictions: folder where the predicted relations are initially stored
        :param final_predictions: folder where the predicted relations along with the original entities are stored
        """
        for filename in os.listdir(self.initial_predictions):
            print(filename)
            if os.stat(self.initial_predictions + filename).st_size == 0:
                continue
            else:
                df = pd.read_csv(self.initial_predictions + filename, header=None, sep="\t")
                df.columns = ['key', 'body']
                df['key'] = df.index + 1
                df['key'] = 'R' + df['key'].astype(str)
                df.to_csv(self.final_predictions + filename, sep='\t', index=False, header=False, mode='a')
