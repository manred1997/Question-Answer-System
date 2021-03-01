from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import os
import argparse

def download_model(outputdir_question_tokenizer:str, outputdir_question_encoder:str, outputdir_ctx_tokenizer:str, outputdir_ctx_encoder:str):
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    print("Save question tokenizer to ", outputdir_question_tokenizer)
    q_tokenizer.save_pretrained(outputdir_question_tokenizer)

    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    print("Save question encoder to ", outputdir_question_encoder)
    q_encoder.save_pretrained(outputdir_question_encoder)

    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    print("Save context tokenizer to ", outputdir_ctx_tokenizer)
    ctx_tokenizer.save_pretrained(outputdir_ctx_tokenizer)

    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    print("Save context encoder to", outputdir_ctx_encoder)
    ctx_encoder.save_pretrained(outputdir_ctx_encoder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_question_tokenizer", default="question_tokenizer", type=str, help="The output directory to download question tokenizer")
    parser.add_argument("--output_question_encoder", default="question_encoder", type=str, help="The output directory to download question encoder")
    parser.add_argument("--output_ctx_tokenizer", default="ctx_tokenizer", type=str, help="The output directory to download ctx tokenizer")
    parser.add_argument("--output_ctx_encoder", default="ctx_encoder", type=str, help="The output directory to download ctx encoder")
    args = parser.parse_args()

    if not os.path.exists(args.output_question_tokenizer): os.makedirs(args.output_question_tokenizer)
    if not os.path.exists(args.output_question_encoder): os.makedirs(args.output_question_encoder)
    if not os.path.exists(args.output_ctx_tokenizer): os.makedirs(args.output_ctx_tokenizer)
    if not os.path.exists(args.output_ctx_encoder): os.makedirs(args.output_ctx_encoder)

    download_model(args.output_question_tokenizer, args.output_question_encoder, args.output_ctx_tokenizer, args.output_ctx_encoder)

if __name__ == "__main__":
    main()