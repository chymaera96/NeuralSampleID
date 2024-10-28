import pandas as pd
import os
import json
import argparse


def query_from_annotation(annot_dir):
    query_dict = {}
    for fname in os.listdir(annot_dir):
        fpath = os.path.join(annot_dir, fname)
        if fpath.endswith('_s.csv'):
            print(f"Processing {fpath}...")
            try:
                df = pd.read_csv(fpath, sep='\t', header=None)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {fpath}")
                continue
            qname = fpath.split('/')[-1].split('_s.csv')[0]
            query_dict[qname] = [[float(df.iloc[0][0]), float(df.iloc[0][0]) + float(df.iloc[0][2])]]
            print(qname, query_dict[qname])

    with open('data/query_dict.json', 'w') as fp:
        json.dump(query_dict, fp)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_dir', type=str, default=None)
    args = parser.parse_args()
    query_from_annotation(args.annot_dir)


if __name__ == "__main__":
    main()
