import json
import os
import re
import sys
import requests
import argparse

def download_data(out_dir: str = None):
    train_data = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")
    if train_data.status_code in (200,):
        with open(os.path.join(out_dir,"train.json"), 'wb') as train_file:
            train_file.write(train_data.content)
    print("Save train data to ",os.path.join(out_dir,"train.json"))
    

    eval_data = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
    if eval_data.status_code in (200,):
        with open(os.path.join(out_dir,"eval.json"), 'wb') as eval_file:
            eval_file.write(eval_data.content)

    print("Save eval data to ",os.path.join(out_dir,"eval.json"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputdir", default="SquadV2_data", type=str, help="The output directory to download file data")

    args = parser.parse_args()
    if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)
    download_data(args.outputdir)

if __name__ == "__main__":
    main()