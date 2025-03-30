
import argparse
import pandas as pd
from topics import get_both_with_qrels

def get_topic_qrels(topics: pd.DataFrame, qrels_dir: str) -> pd.DataFrame:
    qrels = pd.read_csv(qrels_dir, delimiter=r'\s+', header=None, names=['topic_id', 'ignore', 'docno', 'usefulness', 'supportiveness', 'credibility'])
    
    # Get qrels for the relevant topic
    topic_qrels = qrels[qrels['topic_id'].isin(topics['number'])]

    # Filter out Not Useful qrels, and qrels not judged in terms of supportiveness
    relevant_qrels = topic_qrels[(topic_qrels['usefulness'] > 0) & (topic_qrels['supportiveness'] >= 0)]

    return relevant_qrels

def get_topic_id_qrels(topic_ids: list[int] | int, qrels_dir: str) -> pd.DataFrame:
    qrels = pd.read_csv(qrels_dir, delimiter=r'\s+', header=None, names=['topic_id', 'ignore', 'docno', 'usefulness', 'supportiveness', 'credibility'])

    if type(topic_ids) is int:
        topic_ids = [topic_ids]
    
    # Get qrels for specified topics
    topic_qrels = qrels[qrels['topic_id'].isin(topic_ids)]

    # Filter out Not Useful qrels, and qrels not judged in terms of supportiveness
    relevant_qrels = topic_qrels[(topic_qrels['usefulness'] > 0) & (topic_qrels['supportiveness'] >= 0)]

    return relevant_qrels

# NOTE: Sample usage:
# python qrels.py --qrels-dir 'qrels-35topics.txt' --topics-dir 'misinfo-2021-topics.xml' --res-dir 'filtered_qrels.txt' --n 50

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get relevant (useful and with judged supportiveness) qrels for first n topics')

    parser.add_argument('--qrels-dir', type=str, required=True, help='Location of the qrels file')
    parser.add_argument('--topics-dir', type=str, required=True, help='Location of the topics file')
    parser.add_argument('--res-dir', type=str, required=False, help='Location of the result (optional)')
    parser.add_argument('--n', type=int, required=True, help='Number of topics of each type to return')

    args = parser.parse_args()

    topics = get_both_with_qrels(args.n, args.topics_dir, args.qrels_dir)

    res = get_topic_qrels(topics, args.qrels_dir)

    if args.res_dir:
        res.to_csv(args.res_dir, header=None, index=None, sep=' ', mode='a')

    print(res)