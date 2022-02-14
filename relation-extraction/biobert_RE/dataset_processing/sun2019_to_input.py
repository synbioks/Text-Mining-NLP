# convert sun et. al. (2019) chemprot dataset to our input format
import argparse

from utils import utils

def convert(in_filename, out_filename):

    data = utils.read_tsv(in_filename)
    data = data[1:] # remove header

    new_data = []
    for i, (_, _, masked_sent, _, label) in enumerate(data):
        new_data.append([
            str(i),
            label,
            masked_sent,
            'Unavailable'
        ])
    
    utils.save_tsv(out_filename, new_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_filename')
    parser.add_argument('out_filename')
    args = parser.parse_args()

    convert(args.in_filename, args.out_filename)