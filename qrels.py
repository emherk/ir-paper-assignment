
import argparse
import pandas as pd
from topics import get_both

def get_topic_qrels(topics: pd.DataFrame, qrel_dir: str) -> pd.DataFrame:
    qrels = pd.read_csv(qrel_dir, delimiter=r'\s+', header=None, names=['topic_id', 'ignore', 'docno', 'usefulness', 'supportiveness', 'credibility'])
    
    # Get qrels for the relevant topic
    topic_qrels = qrels[qrels['topic_id'].isin(topics['number'])]

    # Filter out Not Useful qrels, and qrels not judged in terms of supportiveness
    relevant_qrels = topic_qrels[(topic_qrels['usefulness'] > 0) & (topic_qrels['supportiveness'] >= 0)]

    return relevant_qrels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get relevant (useful and with judged supportiveness) qrels for first n topics')

    parser.add_argument('--qrels-dir', type=str, required=True, help='Location of the qrels file')
    parser.add_argument('--topics-dir', type=str, required=True, help='Location of the topics file')
    parser.add_argument('--n', type=int, required=True, help='Number of topics of each type to return')

    args = parser.parse_args()

    topics = get_both(args.n, args.topics_dir)

    res = get_topic_qrels(topics, args.qrels_dir)

    print(res)