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

import os
import json

def preprocess_annotations(input_dir, output_file="data/annotations_full.json"):
    """
    Preprocess all annotation files in the input directory and save them to the output directory.
    
    Args:
        input_dir (str): Directory containing the original annotation files.
        output_dir (str): Directory to save the preprocessed annotation files.
    """

    annot_list = []
    # Iterate over all JSON files in the input directory
    for filename in os.listdir(input_dir):
        if not filename.startswith("extra") and filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)

            base_time_annotations = data.get("base_time_annotations", [])

            # Compile the "id" with their corresponding "type"
            id_to_type = {entry["id"]: entry["type"] for entry in base_time_annotations if "type" in entry}

            # Update base_time_annotations to fill missing "type" based on "id"
            for entry in base_time_annotations:
                if "id" in entry and "type" not in entry:
                    entry["type"] = id_to_type.get(entry["id"])

            # Rename keys
            data["query"] = data.pop("base_time_annotations", [])
            data["ref"] = data.pop("sample_time_annotations", [])
            data["query_file"] = data.pop("base_file", "")
            data["ref_file"] = data.pop("sample_file", "")
            

            annot_list.append(data)

    with open(output_file, 'w') as f:
        json.dump(annot_list, f, indent=4)


def generate_query_index(annotations_file, output_file):
    """
    Generate an index file for the dataset class, handling presence and absence annotations.
    Avoids redundant timestamps and skips entries without any absence annotations.
    
    Args:
        annotations_file (str): Path to the annotations_full.json file.
        output_file (str): Path to save the generated index file.
    """
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    index_data = []
    
    for annotation in annotations:
        sample_id = annotation.get("sample_id", "unknown")
        query_file = annotation["query_file"].replace(".mp3", "")
        ref_file = annotation["ref_file"].replace(".mp3", "")
        queries = annotation["query"]
        
        # Separate presence and absence segments
        presence_segments = []
        absence_segments = []
        total_time = max([query["end_time"] for query in queries])
        
        for query in queries:
            start_time = query["start_time"]
            end_time = query["end_time"]
            
            
            if query.get("type") == "absence":
                absence_segments.append((start_time, end_time))
            else:
                presence_segments.append((start_time, end_time))
                index_data.append({
                    "sample_id": sample_id,
                    "query_file": query_file,
                    "ref_file": ref_file,
                    "start_time": start_time,
                    "end_time": end_time
                })
        
        # Skip if there are no absence annotations
        if not absence_segments:
            continue
        
        # Combine presence and absence segments to create negative space
        all_segments = sorted(presence_segments + absence_segments)
        current_time = 0.0
        
        for start, end in all_segments:
            if current_time < start:
                index_data.append({
                    "sample_id": sample_id,
                    "query_file": query_file,
                    "ref_file": ref_file,
                    "start_time": current_time,
                    "end_time": start
                })
            current_time = max(current_time, end)  # Move current_time forward
        
        # Handle the final segment with end_time = -1
        if current_time == total_time:
            index_data.append({
                "sample_id": sample_id,
                "query_file": query_file,
                "ref_file": ref_file,
                "start_time": current_time,
                "end_time": -1
            })
        
        # Remove entries shorter than 1 second
        index_data = [entry for entry in index_data if entry["end_time"] - entry["start_time"] >= 1.0]
        # Sort the index_data by sample_id and start_time
        index_data = sorted(index_data, key=lambda x: (x["sample_id"], x["start_time"]))
    # Save the index file
    with open(output_file, 'w') as f:
        json.dump(index_data, f, indent=4)
    print(f"Index file saved at: {output_file}")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_dir', type=str, default=None)
    args = parser.parse_args()
    # preprocess_annotations(args.annot_dir)
    generate_query_index('data/annotations_full.json', 'data/sample100_query_index.json')


if __name__ == "__main__":
    main()
