import numpy as np
import pandas as pd
from keyBoost_lib.models.yake import yake_extraction
from keyBoost_lib.models.keybert import keybert_extraction
from keyBoost_lib.models.textrank import textrank_extraction
from keyBoost_lib.consensus.statistical import *
from keyBoost_lib.consensus.ranking import rank_consensus

class KeyBoost:

    def __init__(self,
    transformers_model,
    ):
        self.transformers_model = transformers_model


    def extract_keywords(self,
                        text,
                        language,
                        n_top,
                        keyphrases_ngram_max,
                        consensus,
                        models,
                        stopwords=None):



        key_extractions = list()
        # YAKE extraction

        if 'yake' in models:
            yk_rank = yake_extraction(text=text,
                                language=language,
                                keyphrases_ngram_max=keyphrases_ngram_max,
                                n_top=100,
                                stopwords=stopwords)
            key_extractions.append(yk_rank)

        # KeyBERT extraction
        if 'keybert' in models:

            kb_rank = keybert_extraction(text=text,
                                  keyphrases_ngram_max=keyphrases_ngram_max,
                                  n_top=100,
                                  stopwords=stopwords,
                                  transformers_model=self.transformers_model)
            key_extractions.append(kb_rank)

        if 'textrank' in models:

            tr_rank =textrank_extraction(text=text, n_top=100)
            key_extractions.append(tr_rank)

        # Extract scores

        if  consensus   == 'statistical':

            keywords = statistical_consensus(key_extractions=key_extractions,
                                  n_top=n_top)

        elif consensus == 'rank':

            keywords =  rank_consensus(key_extractions=key_extractions,n_top=n_top)


        return keywords
