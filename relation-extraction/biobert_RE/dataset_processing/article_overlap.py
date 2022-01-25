# check how many articles are in both chemprot and drug prot
from os.path import basename
import argparse

from utils import utils

# sanity check, make sure an article is not repeated in a dataset
def add_to_set_check(s, item):
    assert item not in s
    s.add(item)

def main(abs_filename1, abs_filename2):

    article_ids1 = set()
    for line in utils.read_tsv(abs_filename1):
        add_to_set_check(article_ids1, line[0])
    
    article_ids2 = set()
    for line in utils.read_tsv(abs_filename2):
        add_to_set_check(article_ids2, line[0])

    print(f'number of articles in {basename(abs_filename1)}: {len(article_ids1)}')
    print(f'number of articles in {basename(abs_filename2)}: {len(article_ids2)}')
    print(f'number of unique articles in both files: {len(article_ids1.union(article_ids2))}')
    print(f'number of common articles in both files: {len(article_ids1.intersection(article_ids2))}')

if __name__ == '__main__':

    # check number of overlapping articles by article ids
    # deprecated

    parser = argparse.ArgumentParser()
    parser.add_argument('abs_filename1')
    parser.add_argument('abs_filename2')
    args = parser.parse_args()
    main(args.abs_filename1, args.abs_filename2)