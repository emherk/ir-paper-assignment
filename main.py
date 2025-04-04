import pyterrier as pt
import pandas as pd
from ir_measures import define_byquery
from labels import QREL_LABELS
from pathlib import Path
from qrels import get_topic_qrels
from topics import get_both


IDX_DIR = Path('./index').resolve()
TOPICS_DIR = Path('eval\\misinfo-resources-2021\\topics\\misinfo-2021-topics.xml').resolve()
QRELS_DIR = Path('eval\\misinfo-resources-2021\\qrels\\qrels-35topics.txt').resolve()

# https://dl.acm.org/doi/abs/10.1145/3392854
def serp_ms(qrels: pd.DataFrame, run: pd.DataFrame) -> float:
    n = len(run)
    denominator = (n * (n+1)) / 2.0
    
    qrels_indexed = qrels.set_index('doc_id')
    nominator = run.apply(lambda row: serp_ms_x(row, qrels_indexed) * (n - row['rank']), axis=1).sum()

    return nominator / denominator

def serp_ms_x(ranking: pd.Series, qrels_doc_id_indexed: pd.DataFrame):
    # Return 0 if the docno is not part of qrels for this query
    if ranking['doc_id'] not in qrels_doc_id_indexed.index:
        return 0
    
    # Retrieve supportiveness score
    supportiveness = qrels_doc_id_indexed.loc[ranking['doc_id']]['supportiveness']

    # Return misinformation score based on supportiveness and topic stance
    return (supportiveness - 1) if ranking['stance'] == 'helpful' else (1 - supportiveness)

def calculate_qrel_label(qrel: pd.Series, topics: pd.DataFrame) -> int:
    return QREL_LABELS[topics.loc[qrel['qid']]['stance']][qrel['usefulness']][qrel['supportiveness']][qrel['credibility']]

topics = get_both(5, TOPICS_DIR)
qrels = get_topic_qrels(topics, QRELS_DIR)

topics_pt = topics[['number', 'query', 'stance']].rename(columns={'number': 'qid'}).set_index('qid')
qrels_pt = qrels[['topic_id', 'docno', 'usefulness', 'supportiveness', 'credibility']].rename(columns={'topic_id':'qid'})
qrels_pt['label'] = qrels_pt.apply(lambda x: calculate_qrel_label(x, topics_pt), axis=1)
topics_pt = topics_pt.reset_index()

qrels_pt['qid'] = qrels_pt['qid'].astype(str)
topics_pt['qid'] = topics_pt['qid'].astype(str)

SERP_MS = define_byquery(serp_ms, name='SERP-MS')

tfidf = pt.terrier.Retriever(
    str(IDX_DIR),
    wmodel="TF_IDF",
)

bm25 = pt.terrier.Retriever(
    str(IDX_DIR),
    wmodel="BM25",
)

res = pt.Experiment(
    [tfidf, bm25],
    topics_pt,
    qrels_pt,
    names=["TF-IDF", "BM25"],
    eval_metrics=[pt.measures.nDCG @ 10, pt.measures.RR @ 10, pt.measures.MAP, SERP_MS@10]
)

print(res)

