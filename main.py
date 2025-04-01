import pyterrier as pt
import pandas as pd
from ir_measures import define_byquery
from labels import QREL_LABELS
from pathlib import Path
from qrels import get_topic_qrels
from topics import get_both
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


#tfidf + MonoT5
monot5 = (
    tfidf >>
    pt.text.get_text(str(IDX_DIR)) >>
    pt.apply.generic(lambda df: df.rename(columns={"body": "text"})) >>
    monot5_rerank
)

#MonoT5 + BM25
bm25_monot5 = (
    bm25 >>
    pt.text.get_text(str(IDX_DIR)) >>
    pt.apply.generic(lambda df: df.rename(columns={"body": "text"})) >>
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
        ce_scores = ce_model.predict(query_doc_pairs)
        #normalize scores to range from -1 to 1, otherwise the scores will out of range -1 to 1, makes serp-ms not work correctly.
        def normalize_to_range(scores, min_val=-1, max_val=1):
            min_score = np.min(scores)
            max_score = np.max(scores)
            if min_score == max_score:
                return np.full_like(scores, (min_val + max_val) / 2)
            norm_scores = (scores - min_score) / (max_score - min_score)
            return norm_scores * (max_val - min_val) + min_val

        ce_scores_norm = normalize_to_range(np.array(ce_scores), -1, 1)
        df["bert_score"] = ce_scores_norm

        df["score"] = alpha * df["retriever_score"] + (1 - alpha) * df["bert_score"]
        print("bert_score:", df["bert_score"].min(), df["bert_score"].max())
        print("final score:", df["score"].min(), df["score"].max())
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


#Bert + BM25, with weight alpha
bm25_bert_pipeline = pt.apply.generic(rerank(alpha=0.5))
bm25_bert = (
    bm25
    >> pt.text.get_text(str(IDX_DIR))               
    >> pt.apply.generic(lambda df: df.rename(columns={"body": "text"}))
    >> bm25_bert_pipeline
)



#[tfidf, bm25, bert, monot5, bm25_monot5, bm25_bert ],
#["TF-IDF", "BM25","BERT", "MonoT5","BM25+MonoT5","BM25+BERT"],
res = pt.Experiment(
    [tfidf, bm25, bert],
    topics_pt,
    qrels_pt,
    names=["TF-IDF", "BM25","BERT"],
    eval_metrics=[pt.measures.nDCG @ 10, pt.measures.RR @ 10, pt.measures.MAP, SERP_MS@10],
)

print(res)
res.to_csv("experiment_results.csv", index=False)

