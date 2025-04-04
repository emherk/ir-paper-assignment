import pyterrier as pt
import pandas as pd
from ir_measures import define_byquery
from labels import QREL_LABELS
from pathlib import Path
from qrels import get_topic_qrels
from topics import get_both_with_qrels
from pyterrier_t5 import MonoT5ReRanker
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder,SentenceTransformer
import torch
import numpy as np

IDX_DIR = Path('./index').resolve()
TOPICS_DIR = Path('eval/misinfo-resources-2021/topics/misinfo-2021-topics.xml').resolve()
QRELS_DIR = Path('eval/misinfo-resources-2021/qrels/qrels-35topics.txt').resolve()


# https://dl.acm.org/doi/abs/10.1145/3392854
def serp_ms(qrels: pd.DataFrame, run: pd.DataFrame) -> float:
    n = len(run)
    run['final_rank'] = run.reset_index(drop=True).index
    denominator = (n * (n+1)) / 2.0
    
    qrels_indexed = qrels.set_index('doc_id')
    nominator = run.apply(lambda row: serp_ms_x(row, qrels_indexed) * (n - row['final_rank']), axis=1).sum()

    run.drop(columns=['final_rank'])
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

topics = get_both_with_qrels(5, TOPICS_DIR, QRELS_DIR)
qrels = get_topic_qrels(topics, QRELS_DIR)

topics_pt = topics[['number', 'query', 'stance']].rename(columns={'number': 'qid'}).set_index('qid')
qrels_pt = qrels[['topic_id', 'docno', 'usefulness', 'supportiveness', 'credibility']].rename(columns={'topic_id':'qid'})
qrels_pt['label'] = qrels_pt.apply(lambda x: calculate_qrel_label(x, topics_pt), axis=1)
topics_pt = topics_pt.reset_index()

qrels_pt['qid'] = qrels_pt['qid'].astype(str)
topics_pt['qid'] = topics_pt['qid'].astype(str)

SERP_MS = define_byquery(serp_ms, name='SERP-MS')

# Base ranking models
bm25 = pt.terrier.Retriever(
    str(IDX_DIR),
    wmodel="BM25",
)

tfidf = pt.terrier.Retriever(
    str(IDX_DIR),
    wmodel="TF_IDF",
)


#MonoT5
#https://github.com/terrierteam/pyterrier_t5?tab=readme-ov-file#Nogueira21
#https://arxiv.org/pdf/2101.05667
monot5_rerank = MonoT5ReRanker("castorini/monot5-base-msmarco", batch_size=8)

#BM25 + MonoT5
bm25_monot5 = (
    bm25 >>
    pt.text.get_text(str(IDX_DIR)) >>
    monot5_rerank
)


#Bert (Using bert-like model ms-marco-MiniLM-L-6-v2)
#https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def rerank(alpha:float):
    def crossencoder_rerank(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["retriever_score"] = df["score"]  
        query_doc_pairs = list(zip(df["query"], df["text"]))

        df["bert_score"] = ce_model.predict(query_doc_pairs)
        df["score"] = alpha * df["retriever_score"] + (1 - alpha) * df["bert_score"]
        return df
    return crossencoder_rerank

#Bert only (simulate as bert only)  
bert_pipeline = pt.apply.generic(rerank(alpha=0))
bert = (
    bm25
    >> pt.text.get_text(str(IDX_DIR))               
    >> pt.apply.generic(lambda df: df.rename(columns={"body": "text"}))
    >> bert_pipeline
)

#Bert + BM25, with alpha = 0.3
bm25_bert_03_pipeline = pt.apply.generic(rerank(alpha=0.3))
bm25_bert_03 = (
    bm25
    >> pt.text.get_text(str(IDX_DIR))               
    >> pt.apply.generic(lambda df: df.rename(columns={"body": "text"}))
    >> bm25_bert_03_pipeline
)

#Bert + BM25, with alpha = 0.5
bm25_bert_05_pipeline = pt.apply.generic(rerank(alpha=0.5))
bm25_bert_05 = (
    bm25
    >> pt.text.get_text(str(IDX_DIR))               
    >> pt.apply.generic(lambda df: df.rename(columns={"body": "text"}))
    >> bm25_bert_05_pipeline
)

#Bert + BM25, with alpha = 0.7
bm25_bert_07_pipeline = pt.apply.generic(rerank(alpha=0.7))
bm25_bert_07 = (
    bm25
    >> pt.text.get_text(str(IDX_DIR))               
    >> pt.apply.generic(lambda df: df.rename(columns={"body": "text"}))
    >> bm25_bert_07_pipeline
)


# Experiments and result saving
res = pt.Experiment(
    [tfidf, bm25, bert, bm25_bert_03, bm25_bert_05, bm25_bert_07, bm25_monot5],
    topics_pt,
    qrels_pt,
    names=["TF-IDF", "BM25", "BERT(alpha = 0)", "BM25+BERT(alpha = 0.3)", "BM25+BERT(alpha = 0.5)", "BM25+BERT(alpha = 0.7)", "BM25+MonoT5"],
    eval_metrics=[pt.measures.nDCG @ 10, SERP_MS@10],
    baseline=1,
    correction='bonferroni'
)

res.to_csv('results_with_stat.csv', index=False)

res_perquery = pt.Experiment(
    [tfidf, bm25, bert, bm25_bert_03, bm25_bert_05, bm25_bert_07, bm25_monot5],
    topics_pt,
    qrels_pt,
    names=["TF-IDF", "BM25", "BERT(alpha = 0)", "BM25+BERT(alpha = 0.3)", "BM25+BERT(alpha = 0.5)", "BM25+BERT(alpha = 0.7)", "BM25+MonoT5"],
    eval_metrics=[pt.measures.nDCG @ 10, SERP_MS@10],
    perquery=True
)

res_perquery.to_csv("results_perquery_with_stat.csv", index=False)

