from pathlib import Path
import pyterrier as pt
import pandas as pd
from qrels import get_topic_qrels
from topics import get_both


IDX_DIR = Path('./index').resolve()
TOPICS_DIR = Path('eval\\misinfo-resources-2021\\misinfo-resources-2021\\topics\\misinfo-2021-topics.xml').resolve()
QRELS_DIR = Path('eval\\misinfo-resources-2021\\misinfo-resources-2021\\qrels\\qrels-35topics.txt').resolve()

"""
Dictionary for mapping usefulness and supportiveness into labels;
First usefulness and afterwards supportiveness is considered by the dictionary.
Since we're aiming to judge performance on misinformation, we assign labels based
on supportiveness and the stance. Adjust if necessary.
"""
qrel_label_dict = {
    'helpful': {
        2 : {
            2: 1,
            1: 0,
            0: -1,
        },
        1 : {
            2: 1,
            1: 0,
            0: -1,
        }
    },
    'unhelpful': {
        2 : {
            2: -1,
            1: 0,
            0: 1,
        }, 
        1 : {
            2: -1,
            1: 0,
            0: 1,
        }
    }
}


def calculate_qrel_label(qrel: pd.Series, topics: pd.DataFrame) -> int:
    return qrel_label_dict[topics.loc[qrel['qid']]['stance']][qrel['usefulness']][qrel['supportiveness']]

topics = get_both(5, TOPICS_DIR)
qrels = get_topic_qrels(topics, QRELS_DIR)

topics_pt = topics[['number', 'query', 'stance']].rename(columns={'number': 'qid'}).set_index('qid')
qrels_pt = qrels[['topic_id', 'docno', 'usefulness', 'supportiveness']].rename(columns={'topic_id':'qid'})
qrels_pt['label'] = qrels_pt.apply(lambda x: calculate_qrel_label(x, topics_pt), axis=1)
topics_pt = topics_pt.reset_index()

qrels_pt['qid'] = qrels_pt['qid'].astype(str)
topics_pt['qid'] = topics_pt['qid'].astype(str)

tfidf = pt.terrier.Retriever(
    str(IDX_DIR),
    wmodel="TF_IDF",
    num_results=10,
)

bm25 = pt.terrier.Retriever(
    str(IDX_DIR),
    wmodel="BM25",
    num_results=10,
)

res = pt.Experiment(
    [tfidf, bm25],
    topics_pt,
    qrels_pt,
    names=["TF-IDF", "BM25"],
    eval_metrics=[pt.measures.nDCG @ 10, pt.measures.RR @ 10, pt.measures.MAP]
)

print(res)

