import argparse
import pandas as pd


def get_helpful(n: int, dir: str) -> pd.DataFrame:
    data = pd.read_xml(dir)
    return data[data['stance'] == 'helpful'].head(n)

def get_unhelpful(n: int, dir: str) -> pd.DataFrame:
    data = pd.read_xml(dir)
    return data[data['stance'] == 'unhelpful'].head(n)

def get_both(n: int, dir: str) -> pd.DataFrame:
    return pd.concat([get_helpful(n, dir), get_unhelpful(n, dir)])

def get_both_with_qrels(n: int, topics_dir: str, qrels_dir: str) -> pd.DataFrame:
    topics = pd.read_xml(topics_dir)
    qrels = pd.read_csv(qrels_dir, delimiter=r'\s+', header=None, names=['topic_id', 'ignore', 'docno', 'usefulness', 'supportiveness', 'credibility'])

    filtered_topics = topics[topics['number'].isin(qrels['topic_id'])]
    return pd.concat([filtered_topics[filtered_topics['stance'] == 'helpful'].head(n), filtered_topics[filtered_topics['stance'] == 'unhelpful'].head(n)])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get first n helpful and/or unhelpful topics from TREC 2021 health misinfo track')

    parser.add_argument('function', choices=['helpful', 'unhelpful', 'both'], help=['Choose topic types:'])

    parser.add_argument('--topics-dir', type=str, required=True, help='Location of the topics file')
    parser.add_argument('--qrels-dir', type=str, help='Location of the qrels file. If provided returns only topics with existing qrels (only used with both stances)')
    parser.add_argument('--n', type=int, required=True, help='Number of topics of each type to return')

    args = parser.parse_args()

    if args.function == 'helpful':
        res = get_helpful(args.n, args.topics_dir)
    elif args.function == 'unhelpful':
        res = get_unhelpful(args.n, args.topics_dir)
    elif args.function == 'both':
        if args.qrels_dir:
            res = get_both_with_qrels(args.n, args.topics_dir, args.qrels_dir)
        else:
            res = get_both(args.n, args.topics_dir)

    print(res)