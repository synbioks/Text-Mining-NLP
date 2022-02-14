cpr_map = {
    'PART-OF': 'CPR-1',

    'REGULATOR': 'CPR-2',
    'DIRECT-REGULATOR': 'CPR-2',
    'INDIRECT-REGULATOR': 'CPR-2',

    'UPREGULATOR': 'CPR-3',
    'ACTIVATOR': 'CPR-3',
    'INDIRECT-UPREGULATOR': 'CPR-3',

    'DOWNREGULATOR': 'CPR-4',
    'INHIBITOR': 'CPR-4',
    'INDIRECT-DOWNREGULATOR': 'CPR-4',

    'AGONIST': 'CPR-5',
    'AGONIST-ACTIVATOR': 'CPR-5',
    'AGONIST-INHIBITOR': 'CPR-5',

    'ANTAGONIST': 'CPR-6',

    'MODULATOR': 'CPR-7',
    'MODULATOR-ACTIVATOR': 'CPR-7',
    'MODULATOR-INHIBITOR': 'CPR-7',

    'COFACTOR': 'CPR-8',

    'SUBSTRATE': 'CPR-9',
    'PRODUCT-OF': 'CPR-9',
    'SUBSTRATE_PRODUCT-OF': 'CPR-9',

    'NOT': 'NOT',
    'UNDEFINED': 'NOT' # there is only two UNDEFINED in the dataset, we will ignore them
}

cpr_label_id = {
    'CPR-1': 0,
    'CPR-2': 1,
    'CPR-3': 2,
    'CPR-4': 3,
    'CPR-5': 4,
    'CPR-6': 5,
    # CPR-7 and 8 are not used because they have too few samples
    'CPR-9': 6,
    'NOT': 7
}

cpr_id_label = {v: k for k, v in cpr_label_id.items()}

chemprot_sun2019_label_id = {
    'CPR:3': 0,
    'CPR:4': 1,
    'CPR:9': 2,
    'false': 3,
    # CPR 5 and 6 are ignored
}

# get label map, if name is not specified, return the most recently used
# TODO: make this into a decorator
cached_choice = None
def get_label_map(label_map_name=None):
    global cached_choice
    if label_map_name is None:
        pass # do nothing
    elif label_map_name == 'merged':
        cached_choice = cpr_label_id
    elif label_map_name == 'chemprot_sun2019':
        cached_choice = chemprot_sun2019_label_id
    else:
        assert False, f'unknown label map name {label_map_name}'
    return cached_choice
