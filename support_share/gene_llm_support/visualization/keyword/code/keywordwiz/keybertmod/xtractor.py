##############################################
# created on: 11/16/2023
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
import argparse
from transformers import *
from keybert import KeyBERT
import sys


def xtract_keyword(bertBackBone, genellmsentence, args):
        keywords=None
        ngram=(args.ngram_min,args.ngram_max)
        if args.keybert_type == "basic":
                try:
                        kw_model = KeyBERT(bertBackBone)
                        keywords = kw_model.extract_keywords(genellmsentence,highlight=True)
                except:
                        print(f"Not able to run {args.keybert_type} keybert")
        elif args.keybert_type == "phrase":
                try:
                        kw_model = KeyBERT(bertBackBone)
                        keywords = kw_model.extract_keywords(genellmsentence,highlight=True, keyphrase_ngram_range = ngram, stop_words=args.stop_words)
                except:
                        print(f"Not able to run {args.keybert_type} keybert")                
        elif args.keybert_type == "maxdis":
                        kw_model = KeyBERT(bertBackBone)
                        keywords = kw_model.extract_keywords(genellmsentence,highlight=True, keyphrase_ngram_range= ngram, stop_words=args.stop_words, use_maxsum=True, nr_candidates=args.nr_candidate, top_n= args.n_top)       
        elif args.keybert_type == "maxrelavence":
                try:
                        kw_model = KeyBERT(bertBackBone)
                        keywords = kw_model.extract_keywords(genellmsentence, highlight=True, keyphrase_ngram_range=ngram, stop_words=args.stop_words, use_mmr=True, diversity=args.diversity)
                except:
                        print(f"Not able to run {args.keybert_type} keybert")  
        else:
                sys.exit('Unknown type for keybert')
        
        return keywords
