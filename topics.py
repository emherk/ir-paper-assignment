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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get first n helpful and/or unhelpful topics from TREC 2021 health misinfo track')

    parser.add_argument('function', choices=['helpful', 'unhelpful', 'both'], help=['Choose topic types:'])

    parser.add_argument('--topics-dir', type=str, required=True, help='Location of the topics file')
    parser.add_argument('--n', type=int, required=True, help='Number of topics of each type to return')

    args = parser.parse_args()

    if args.function == 'helpful':
        res = get_helpful(args.n, args.topics_dir)
    elif args.function == 'unhelpful':
        res = get_unhelpful(args.n, args.topics_dir)
    elif args.function == 'both':
        res = get_both(args.n, args.topics_dir)

    print(res)