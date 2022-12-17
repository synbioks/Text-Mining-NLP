# Author: Cora Lewis for RelEx
from data_prep import Dataset
from segment import Segmentation
from func import file
import os
import numpy as np
from csv import reader


class Set_Connection:
    def __init__(self, dataset=None, rel_labels=None, no_labels=None, test=False, no_rel_multiple=False,
                 dominant_entity = 'S', no_of_cores = 64, down_sample=None, down_sample_ratio=0.2, predictions_folder = None,
                 write_Entites = False, sentence_only = False):
        """
        Creates data objects directly from the dataset folder and call for segmentation or take in segments (a set of CSVs)
        :type write_Entites: write entities and predictions to file
        :param sentence_only: flag when we need to consider segments for sentence CNN only
        :param track: path to track information (file, first entity, second entity)
        :param dataset: path to dataset
        :param rel_labels: list of entities that create the relations
        :param no_labels: name the label when entities that do not have relations in a sentence are considered
        :param no_rel_multiple: flag whether multiple labels are possibles for No-relation
        :param CSV: flag to decide to read from the CSVs directly
        :param test: flag to run test-segmentation options
        :param no_of_cores: no of cores to run the parallelized segmentation
        :param predictions_folder: path to predictions (output) folder
        """
        self.sentence_only = sentence_only
        self.test = test

        # data -brat format(no CSVs): run Segmentation module to get the segmentation object
        self.dataset = Dataset(dataset)
        self.rel_labels = rel_labels
        self.no_labels = no_labels
        self.no_rel_multiple = no_rel_multiple
        self.data_object = Segmentation(self.dataset, self.rel_labels, self.no_labels, self.no_rel_multiple, test=self.test, dominant_entity = dominant_entity, no_of_cores=no_of_cores, predictions_folder = predictions_folder,write_Entites = write_Entites, down_sample=down_sample, down_sample_ratio=down_sample_ratio  ).segments
