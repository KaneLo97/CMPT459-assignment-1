import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# np.set_printoptions(threshold=sys.maxsize)

input_df = pd.read_json('test.json.zip')
# input_df.head(10).to_csv("test.csv")

# Which columns seem useful for text feature extraction?
    # Features -> Not much procesing needed.
        # Features is tricky because of nested list of features
    # Discription -> 
        # 1) tokenizing
        # 2) counting
        # 3) normalizing

# print(input_df['features'])
descriptions = input_df['description']
features = input_df['features']
# print(features)

vectorizer_des = CountVectorizer()
corpus_des = descriptions.to_numpy()
# print(corpus_des)
X_des = vectorizer_des.fit_transform(corpus_des)
print(X_des.toarray())

# TOKENIZER IN CountVectorizer expects string or bytes-like object
# [['Doorman, Elevator, Cats Allowed, Dogs Allowed'],['Pre-War, Dogs Allowed, Cats Allowed']] WON'T WORK
# Esentially, we need to convert to ['Doorman, Elevator, Cats Allowed, Dogs Allowed','Pre-War, Dogs Allowed, Cats Allowed']

# This does essentially what is described above
sample_input = [['Doorman, Elevator, Cats Allowed, Dogs Allowed'],['Pre-War, Dogs Allowed, Cats Allowed']]
input_corrected = [" ".join(x) for x in sample_input]
# print(input_corrected)

# vectorizer_fea = CountVectorizer()
# corpus_fea = [" ".join(x) for x in features]
# print(corpus_fea)
# X_fea = vectorizer_fea.fit_transform(corpus_fea)
# print(vectorizer_fea.get_feature_names())
# print(X_fea.toarray())