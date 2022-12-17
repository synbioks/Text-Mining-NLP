# Author : Samantha Mahendran for RelEx-GCN
import numpy as np
from data_prep import Annotation
from func import file, normalization


def collect_entities(concept1_list, concept2_list, appeared=None):
    """
    list the entities
    :param concept1_list: entity 1 segment
    :param concept2_list: entity 2 segment
    :param appeared: existing entity list
    :return:
    """
    if appeared is None:
        appeared = []
    for i in range(len(concept1_list)):
        entity1 = concept1_list[i]

        if entity1 in appeared:
            continue
        else:
            appeared.append(entity1)
    for i in range(len(concept2_list)):
        entity2 = concept2_list[i]

        if entity2 in appeared:
            continue
        else:
            appeared.append(entity2)
    return appeared


def label_entity_pair(data, data_path, track_info):
    """
    To replace the entity pair with the semantic type / label of the  entities
    :param data: sentences
    :param data_path: path to ann files
    :param track_info: filename and entity information
    :return:
    """
    updated_data = []
    for i in range(len(data)):
        sentence = data[i]

        file_path = data_path + str(track_info[i][0]) + ".ann"
        ann_obj = Annotation(file_path)
        entity1 = ann_obj.annotations['entities']["T" + track_info[i][1]][3]
        entity2 = ann_obj.annotations['entities']["T" + track_info[i][2]][3]

        e1_label = ann_obj.annotations['entities']["T" + track_info[i][1]][0]
        e2_label = ann_obj.annotations['entities']["T" + track_info[i][2]][0]

        e1 = normalization.remove_Punctuation(str(entity1).strip())
        sentence = sentence.replace(e1, "@" + e1_label + "$")
        e2 = normalization.remove_Punctuation(str(entity2).strip())
        sentence = sentence.replace(e2, "@" + e2_label + "$")

        updated_data.append(sentence)
    return updated_data


def mask_entities(data, data_path, track_info):
    """
    To mask other entities except for the ones that make the relation
    :param data: sentences
    :param data_path: path to ann files
    :param track_info: filename and entity information
    """
    updated_data = []
    for i in range(len(data)):
        sent1 = data[i]
        all_entities = set()
        entity_pair = set()
        remove_entities = set()
        entity_pair.add(track_info[i][1])
        entity_pair.add(track_info[i][2])

        for j in range(len(data)):
            sent2 = data[j]
            if i == j:
                continue
            elif sent1 == sent2 and str(track_info[i][0]) == str(track_info[j][0]):
                file_path = data_path + str(track_info[i][0]) + ".ann"
                ann_obj = Annotation(file_path)

                all_entities.add(track_info[j][1])
                all_entities.add(track_info[j][2])
                # break
            else:
                sentence = sent1
        remove_entities = all_entities - entity_pair

        if len(remove_entities) != 0:
            sentence = sent1
            for entity in remove_entities:
                e = ann_obj.annotations['entities']["T" + entity][3]
                e = normalization.remove_Punctuation(str(e).strip())
                sentence = sentence.replace(e, 'X')

        updated_data.append(sentence)
    return updated_data


class Model:

    def __init__(self, data_object, data_object_test=None, segment=False, entity_masking=False,
                 replace_entity_pair=True,
                 test=False, write_predictions=False, generalize=False, train_path=None, test_path=None):
        """
        :param data_object: training data object
        :param data_object_test: testing data object (None -during 5 CV)
        :param segment: Flag to be set to activate segment-CNN (default-False)
        :param test: Flag to be set to validate the model on the test dataset (default-False)
        :type write_predictions: write entities and predictions to file
        :param generalize: flag when relations are not dependent on the first given relation label

        """
        self.entity_masking = entity_masking
        self.write_Predictions = write_predictions
        self.generalize = generalize
        self.test = test
        self.segment = segment
        self.data_object = data_object
        self.data_object_test = data_object_test

        # read dataset from external files
        train_data = data_object['sentence']

        self.train_label = data_object['label']
        # tracks the entity pair details for each relation
        self.train_track = np.asarray(data_object['track']).reshape((-1, 3)).tolist()
        train_concept1 = data_object['seg_concept1']
        train_concept2 = data_object['seg_concept2']
        train_entities = collect_entities(train_concept1, train_concept2)

        # to read in segments
        if self.segment:
            train_preceding = data_object['seg_preceding']
            train_middle = data_object['seg_middle']
            train_succeeding = data_object['seg_succeeding']
            merged = []
            for i in range(len(train_concept1)):
                merged_seg = train_concept1[i] + train_middle[i] + train_concept2[i]
                merged.append(merged_seg)
            train_data = merged
            f = open('merged_middle_train.txt', 'w')
            f.write('\n'.join(train_data))
            f.close()

        # test files only
        if self.test:
            test_data = data_object_test['sentence']
            self.y_test = data_object_test['label']
            test_track = data_object_test['track']
            self.test_track = np.asarray(test_track).reshape((-1, 3)).tolist()
            test_list = ['test'] * len(self.y_test)

            # tracks the entity pair details for a relation
            test_concept1 = data_object_test['seg_concept1']
            test_concept2 = data_object_test['seg_concept2']
            self.unique_entities = collect_entities(test_concept1, test_concept2, train_entities)

            # to read in segments
            if segment:
                test_preceding = data_object_test['seg_preceding']
                test_middle = data_object_test['seg_middle']
                test_succeeding = data_object_test['seg_succeeding']
                merged = []
                for i in range(len(test_concept1)):
                    merged_seg = test_concept1[i] + test_middle[i] + test_concept2[i]
                    merged.append(merged_seg)
                test_data = merged
                f = open('merged_middle_test.txt', 'w')
                f.write('\n'.join(test_data))
                f.close()
        else:
            # when running only with train data
            test_data = None
            test_labels = None

        if entity_masking:
            train_data = mask_entities(train_data, train_path, self.train_track)
            test_data = mask_entities(test_data, test_path, self.test_track)

        if replace_entity_pair:
            train_data = label_entity_pair(train_data, train_path, self.train_track)
            test_data = label_entity_pair(test_data, test_path, self.test_track)

        # creating metalist for training data
        meta_train_list = []
        for i in range(len(self.train_label)):
            meta = train_data[i] + '\t' + self.train_label[i] + '\t' + str(self.train_track[i])
            meta_train_list.append(meta)
        self.meta_train_data = meta_train_list

        # if test data is present
        if self.test:
            # creating metalist for test data
            meta_test_list = []
            for i in range(len(self.y_test)):
                meta = test_data[i] + '\t' + self.y_test[i] + '\t' + str(self.test_track[i])
                meta_test_list.append(meta)
        self.meta_test_data = meta_test_list
