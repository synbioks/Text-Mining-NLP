import os
from argparse import ArgumentParser

REPLACE_PATTERN = '[[[[REPLACE]]]]'

def get_replace_pattern(pos):
    return f'[[[[ARG:{pos}]]]]'

def gen_yml_file(template, arguments):
    for pos, arg in enumerate(arguments):
        template = template.replace(get_replace_pattern(pos), arg)
    return template

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--yml-temp', type=str, required=True)
    parser.add_argument('--yml-args', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    template = None
    with open(args.yml_temp, 'r') as fin:
        template = fin.read()
    
    # arguments, currently there isn't any escaping option use with caution
    generated_ymls = []
    with open(args.yml_args, 'r') as fin:
        for line in fin:
            yml_args = line.strip().split('\t')
            generated_ymls.append(gen_yml_file(template, yml_args))
    
    for i, generated_yml in enumerate(generated_ymls):
        # file name is defined by the name of the template file and the name of the argument file
        output_filename = f'{os.path.basename(args.yml_temp)}_{os.path.basename(args.yml_args)}_{i+1}.yml'
        output_filename = os.path.join(args.output, output_filename)
        with open(output_filename, 'w') as fout:
            fout.write(generated_yml)