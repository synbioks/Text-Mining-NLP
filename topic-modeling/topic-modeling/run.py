import sys
import json
import pickle

sys.path.insert(0, 'src')
from features.data_words import save_data_words
from models.models import run_model
from util import *


def load_params(fp):
    """
    Load params from json file 
    """
    with open(fp) as fh:
        param = json.load(fh)

    return param

def main(targets):
    """
    Runs the main project pipeline logic, given the target 
    targets must contain: 'baseline_df' ...
    """
    if 'pub_year' in targets:
        params = load_params('config/pub_year.json')
        extract_pub_year(**params)

    if 'create_data_words' in targets:
        params = load_params('config/data_words.json')
        save_data_words(**params)

    if 'model' in targets:
        params = load_params('config/model.json')
        run_model(**params)
    

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)