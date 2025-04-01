# Script taken from https://trec-health-misinfo.github.io/2021.html
"""
Script to add docnos to files in c4/no.clean
To process all files:
python docnos.py --path <path-to-c4-repo>
To process a subset, e.g. the first 20 files:
python docnos.py --path <path-to-c4-repo> --pattern 000[01]?
"""
# import argparse
# import glob
# import gzip

# parser = argparse.ArgumentParser(description='Add docnos to C4 collection.')
# parser.add_argument('--path', type=str, help='Root of C4 git repo.', required=True)
# parser.add_argument('--pattern', type=str, default="?????", help='File name patterns to process.')
# args = parser.parse_args()
# pattern = args.pattern
# path = args.path


# def new_docno(file_number, line_number):
#     return f'en.noclean.c4-train.{file_number}-of-07168.{line_number}'


# files = sorted(list(glob.iglob(f'{path}/en.noclean/c4-train.{pattern}-of-07168.json.gz')))

# for filepath in files:
    # with gzip.open(filepath) as f:
    #     file_number = filepath[-22:-22 + 5]
    #     file_name = filepath[-31:]
    #     print(f"adding docnos to file number {file_number} ...")
    #     with gzip.open(f'{path}/en.noclean.withdocnos/{file_name}', 'wb') as o:
    #         for line_number, line in enumerate(f.readlines()):
    #             line = line.decode('utf-8')
    #             new_line = f"{{\"docno\":\"{new_docno(file_number, line_number)}\",{line[1:]}"
    #             o.write(new_line.encode('utf-8'))

import argparse
import gzip
import os
import random
import subprocess
from itertools import islice
import time
import numpy as np
from topics import get_both_with_qrels
from tqdm import tqdm
from qrels import get_topic_qrels


def parse_file_number(docno: str) -> str:
    return docno[20:25]

def parse_line_number(docno: str) -> int:
    return int(docno[35:])

def new_docno(file_number, line_number):
    return f'en.noclean.c4-train.{file_number}-of-07168.{line_number}'

parser = argparse.ArgumentParser(description='Download and add docnos to documents given the qrels')
parser.add_argument('--c4-dir', type=str, required=True, help='Root of C4 git repo.')
parser.add_argument('--topics-dir', type=str, required=True, help='Location of the topics file')
parser.add_argument('--qrels-dir', type=str, required=True, help='Location of the qrels file')
parser.add_argument('--n', type=int, required=True, help='Number of topics of each type to get')
parser.add_argument('--verbose', action='store_true', help='Prints command line excecution info')
args = parser.parse_args()

topics = get_both_with_qrels(args.n, args.topics_dir, args.qrels_dir)
qrels = get_topic_qrels(topics, args.qrels_dir)

qrels['fileno'] = qrels['docno'].map(parse_file_number)
qrels['lineno'] = qrels['docno'].map(parse_line_number)

os.makedirs('data', exist_ok=True)
cwd = os.getcwd()
datapath = os.path.join(cwd, 'data')
os.chdir(args.c4_dir)
random.seed(42)

for fileno in tqdm(np.sort(qrels['fileno'].unique()), desc='Document download progress: ', unit='file'):
    filepath = f'en.noclean/c4-train.{fileno}-of-07168.json.gz'

    res = subprocess.run(['git', 'lfs', 'pull', '--include', filepath], check=True, capture_output=True, text=True)
    if args.verbose:
        print(res)

    with gzip.open(filepath) as f:
        file_number = filepath[-22:-22 + 5]
        file_name = filepath[-31:]
        if args.verbose:
            print(f"adding docnos to file number {file_number} ...")

        # Subsample docs with qrels
        relevant_docs = qrels[qrels['fileno'] == fileno]['lineno'].values
        other_docs = [n for n in range(relevant_docs.max()) if n not in relevant_docs]
        # Additionaly subsample 10 random docs for each document with qrels
        required_docs = np.concat((relevant_docs, random.sample(other_docs, len(relevant_docs)*10)))

        with gzip.open(os.path.join(datapath, file_name), 'wb') as o:
            for line_number in required_docs:
                f.seek(0)
                line = next(islice(f, line_number, line_number+1), None).decode('utf-8')
                new_line = f"{{\"docno\":\"{new_docno(file_number, line_number)}\",{line[1:]}"
                o.write(new_line.encode('utf-8'))

    subprocess.run(['rm', '-rf', filepath])
    subprocess.run(['rm', '-rf', './.git/lfs/objects'])

os.chdir(cwd)