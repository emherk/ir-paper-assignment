"""
Script to create a pyterrier index in c4 for the TREC misinformation 2021 dataset
Before running this script, run the script for downloading the documents (docnos.py)
When running specify the directory of the data (documents):
python index.py --data-dir <path-to-data-folder>
"""
import argparse
import glob
import gzip
import json
from pathlib import Path
import pyterrier as pt


def index_documents(data_dir: str) -> None:
    files = sorted(list(glob.iglob(f'{data_dir}/c4-train.?????-of-07168.json.gz')))

    def document_iterator():
        for filename in files:
            with gzip.open(filename) as f:
                for line in f:
                    data = json.loads(line)
                    yield {'docno': data['docno'], 'text': data['text']}

    index_path = Path('./index').resolve()
    indexer = pt.IterDictIndexer(
        str(index_path),
        meta={
            'docno': 42
        }
    )
    index_ref = indexer.index(document_iterator())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index the C4 collection')
    parser.add_argument('--data-dir', type=str, required=True, help='Location of the files to be indexed')
    args = parser.parse_args()

    index_documents(args.data_dir)