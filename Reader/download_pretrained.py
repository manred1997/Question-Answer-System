from transformers import ElectraTokenizer, ElectraForQuestionAnswering
import os
import argparse

def download_model(outputdir_tokenizer: str, outputdir_pretrained: str):
    slow_tokenizer = ElectraTokenizer.from_pretrained("bert-base-uncased")
    print("Save tokenizer to ", outputdir_tokenizer)
    slow_tokenizer.save_pretrained(outputdir_tokenizer)

    model = ElectraForQuestionAnswering.from_pretrained("google/electra-base-discriminator")
    model.save_pretrained(outputdir_pretrained)
    print("Save model electra pretrained to", outputdir_pretrained)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_tokenizer", default="electra_base_uncased", type=str, help="The output directory to download tokenizer")
    parser.add_argument("--output_pretrained", default="electra_QA", type=str, help="The output directory to download file model ElectraForQuestionAnswering")
    args = parser.parse_args()

    if not os.path.exists(args.output_tokenizer): os.makedirs(args.output_tokenizer)
    if not os.path.exists(args.output_pretrained): os.makedirs(args.output_pretrained)

    download_model(args.output_tokenizer, args.output_pretrained)

if __name__ == "__main__":
    main()