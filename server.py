import streamlit as stl
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from keyBoost.keyBoost import *
from PIL import Image
import spacy

image = Image.open('keyboost.png')


col1, col2, col3 = stl.beta_columns([6,12,1])

with col1:
    stl.write("")

with col2:
    stl.image(image)

with col3:
    stl.write("")

initial_text = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs.[1] It infers a
         function from labeled training data consisting of a set of training examples.[2]
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias).
      """


keyboost = KeyBoost(transformers_model='distilbert-base-nli-mean-tokens')

language = stl.selectbox(label='What is the language of the text ?',
                          options =['en','fr'])


if language == 'en':

    nlp = spacy.load('en_core_web_sm')
    stopwords = nlp.Defaults.stop_words



elif language == 'fr':

    nlp = spacy.load('fr_core_news_sm')
    stopwords = nlp.Defaults.stop_words



selected_models = stl.multiselect(label='What are the underlying extraction models you want to use ?',
                          default=['yake','textrank','keybert'],
                          options =['yake','textrank','keybert'])


selected_consensus = stl.selectbox(label='What kind of consensus do you want to be performed ?',
                          options =['statistical','rank'])


txt = stl.text_area(label='Text to analyze',
                   value=initial_text,
                   height=350)

n_top = stl.slider(label='How many keywords at most do you want to extract ?',
                        min_value=1,max_value=10,value=5)





with stl.spinner(text='Wait for it...'):
    if txt == '':
        stl.info('Please input some text in the dedicated area')

    else:

        keywords = keyboost.extract_keywords(text=txt,
                               language=language,
                               n_top=n_top,
                               keyphrases_ngram_max=2,
                               stopwords=stopwords,
                               models = selected_models,
                               consensus = selected_consensus)

        if 'textrank' in selected_models:
           keywords = [k for k in keywords if k not in stopwords]

        css = '''text-transform: lowercase;
        	background: linear-gradient(to right, #acb4fc 0%, #6fd4fc 100%);
        	-webkit-text-fill-color: white;
        	display: inline-block;
            padding: 3px 3px;
            margin: 5px 5px;'''

        # maybe adding the score feature later
        # if keyboost.is_statistical_consensus_completed:
        #     mkds = ''
        #
        #
        #
        #
        #     css_confidence = '''text-transform: lowercase;
        #     	background: black
        #     	-webkit-text-fill-color: white;
        #     	display: inline-block;
        #         padding: 3px 3px;
        #         margin: 5px 5px;'''
        #
        #     max_score = keyboost.statistical_consensus_scores['Score'].max()
        #
        #     for k,s in keyboost.statistical_consensus_scores.values:
        #         print(k,s)
        #         mkds+='''<p style='{}'>{} (score:{})</p>'''.format(css,k,round(s/max_score*100,2))
        #     stl.markdown(mkds,unsafe_allow_html=True)
        # else:

        mkd = ''
        for k in keywords:
            mkd+='''<p style='{}'>{}</p>'''.format(css,k)

        stl.markdown(mkd,unsafe_allow_html=True)
