from pathlib import Path
import json

import pandas as pd

from topics import get_both_with_qrels
from qrels import get_topic_qrels
from labels import QREL_LABELS

TOPICS_DIR = Path('../misinfo-2021/topics/misinfo-2021-topics.xml').resolve()
QRELS_DIR = Path('../misinfo-2021/qrels/qrels-35topics.txt').resolve()

topics = get_both_with_qrels(5, TOPICS_DIR, QRELS_DIR)
qrels = get_topic_qrels(topics, QRELS_DIR)

def calculate_qrel_label(qrel: pd.Series, topics: pd.DataFrame) -> int:
    return QREL_LABELS[topics.loc[qrel['qid']]['stance']][qrel['usefulness']][qrel['supportiveness']][qrel['credibility']]

topics_pt = topics[['number', 'query', 'stance']].rename(columns={'number': 'qid'}).set_index('qid')
qrels_pt = qrels[['topic_id', 'docno', 'usefulness', 'supportiveness', 'credibility']].rename(columns={'topic_id':'qid'})
qrels_pt['label'] = qrels_pt.apply(lambda x: calculate_qrel_label(x, topics_pt), axis=1)
topics_pt = topics_pt.reset_index()

qrels_pt['qid'] = qrels_pt['qid'].astype(str)
topics_pt['qid'] = topics_pt['qid'].astype(str)

def qrels_misinfo_scores(qrels_doc_id_indexed: pd.DataFrame, topics_indexed):
    misinfo_labels_per_qid = {}
    for idx in topics_indexed.index:
        misinfo_labels_per_qid[idx] = {'misinfo': 0, 'neutral': 0, 'debunks': 0}

    for idx, qrel in qrels_doc_id_indexed.iterrows():
        supportiveness = qrel['supportiveness']
        score = (supportiveness - 1) if topics_indexed.loc[[qrel['qid']]]['stance'].values[0] == 'helpful' else (1 - supportiveness)

        if score == -1:
            key = 'misinfo'
        elif score == 0:
            key = 'neutral'
        else:
            key = 'debunks'

        misinfo_labels_per_qid[qrel['qid']][key] += 1

    return misinfo_labels_per_qid


if __name__ == '__main__':
    qrels_indexed = qrels.set_index('docno')
    topics_indexed = topics_pt.set_index('qid')

    misinfo_labels_per_qid = qrels_misinfo_scores(qrels_pt, topics_indexed)

    with open('misinfo_labels_per_qid.json', 'w') as f:
        json.dump(misinfo_labels_per_qid, f)

    total_misinfo = sum(d['misinfo'] for d in misinfo_labels_per_qid.values())
    total_debunks = sum(d['debunks'] for d in misinfo_labels_per_qid.values())
    # Because for each qrel we had we additionally sampled 10 random docs
    total_neutral = 11 * len(qrels_indexed) - total_misinfo - total_debunks

    print(f'Total misinfo: {total_misinfo}')
    print(f'Total debunks: {total_debunks}')
    print(f'Total neutral: {total_neutral}')
