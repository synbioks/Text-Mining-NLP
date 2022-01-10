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