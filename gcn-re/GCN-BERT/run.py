# Author : Samantha Mahendran for RelEx
import ast
import configparser

import train_test
from func import re_number, extract_entites, CLEF_convert
# read the config file
config = configparser.ConfigParser()
config.read('configs/drugprot.ini')

# read parameters from the config file
if config.getboolean('SEGMENTATION', 'no_relation'):
    no_rel_label = ast.literal_eval(config.get("SEGMENTATION", "no_rel_label"))
else:
    no_rel_label = None

test = config.getboolean('DEFAULT', 'test')
binary = config.getboolean('DEFAULT', 'binary_classification')
write_predictions = config.getboolean('DEFAULT', 'write_predictions')
write_no_relations = config.getboolean('PREDICTIONS', 'write_no_relations')
downsample_allow = config.getboolean('SEGMENTATION', 'downsample_allow')
rel_labels = ast.literal_eval(config.get("SEGMENTATION", "rel_labels"))
entity_masking = config.getboolean('GCN_MODELS', 'entity_masking')
replace_entity_pair = config.getboolean('GCN_MODELS', 'replace_entity_pair')
segment = config.getboolean('SEGMENTATION', 'segment')
dominant_entity = ast.literal_eval(config.get("SEGMENTATION", "dominant_entity"))
# train-test
if test:
    # binary classification
    if binary:
        print("Please note if it is binary classification predictions must be written to files")
        # write entities to the output files
        extract_entites.write_entities(config['SEGMENTATION']['test_path'], config['PREDICTIONS']['final_predictions'], config['PREDICTIONS']['binary_predictions'])
        # for each label
        for label in rel_labels[1:]:
            rel_labels = [rel_labels[0], label]
            # perform segmentation
            seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'],
                                                     config['SEGMENTATION']['test_path'], rel_labels, no_rel_label,
                                                     config.getboolean('SEGMENTATION', 'no_rel_multiple'), dominant_entity[0],
                                                     config.getint('SEGMENTATION', 'no_of_cores'))
        #    run GCN model
            train_test.run_GCN_model(seg_train, seg_test, config['GCN_MODELS']['embedding_path'],
                                     config.getint('GCN_MODELS', 'embedding_dim'),
                                     config['GCN_MODELS']['model'], config['GCN_MODELS']['window_size'], dominant_entity[0], segment,
                                     entity_masking, replace_entity_pair,
                                     write_predictions, write_no_relations,
                                     config['PREDICTIONS']['initial_predictions'],
                                     config['PREDICTIONS']['binary_predictions'], config['SEGMENTATION']['train_path'],
                                     config['SEGMENTATION']['test_path'])

        re_number.append(config['PREDICTIONS']['binary_predictions'], config['PREDICTIONS']['final_predictions'])

    # multi-class classification
    else:
        extract_entites.write_entities(config['SEGMENTATION']['test_path'], config['PREDICTIONS']['final_predictions'], config['PREDICTIONS']['binary_predictions'])
        seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'],
                                                 config['SEGMENTATION']['test_path'], rel_labels, no_rel_label,
                                                 config.getboolean('SEGMENTATION', 'no_rel_multiple'),
                                                 dominant_entity[0],
                                                 config.getint('SEGMENTATION', 'no_of_cores'),
                                                )
        train_test.run_GCN_model(seg_train, seg_test, config['GCN_MODELS']['embedding_path'],
                                 config.getint('GCN_MODELS', 'embedding_dim'),
                                 config['GCN_MODELS']['model'], config['GCN_MODELS']['window_size'], dominant_entity[0], segment,
                                     entity_masking, replace_entity_pair,
                                 write_predictions, write_no_relations,
                                 config['PREDICTIONS']['initial_predictions'],
                                 config['PREDICTIONS']['binary_predictions'], config['SEGMENTATION']['train_path'],
                                 config['SEGMENTATION']['test_path'])
        re_number.append(config['PREDICTIONS']['binary_predictions'], config['PREDICTIONS']['final_predictions'])
        #