import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys 

# Settings for Terminal 
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# np.set_printoptions(threshold=sys.maxsize)

input_df = pd.read_json('train.json.zip')

# input_df.head(10).to_csv("test.csv")

# print(input_df['features'])
descriptions = input_df['description']
features = input_df[['features', 'description', 'listing_id']]
# print(features)

# # # # # # # # # # # # # # # # # # # # # # # # 
# Fill in Missing Features using Description  # 
# # # # # # # # # # # # # # # # # # # # # # # # 

mask = features['features'].apply(len) == 0
missing_features_df = features[mask]
# new_df.to_csv('check.csv')

# Build a list of features in existing data
df_has_features = features[~mask]
all_lists = df_has_features['features'].to_numpy()
feature_list = set()
for each_list in all_lists:
    for elem in each_list:
        feature_list.add(elem)
# print(feature_list)

# Parse the description column and find matching features
def fill_features(row):
    result_list = []
    for feature in feature_list:
        if feature in row:
            # print(row, " has feature: ", feature)
            result_list.append(feature)
    # print(result_list)
    return result_list

# Apply the function to fill missing features
missing_features_df['features'] = missing_features_df.apply(lambda row: fill_features(row['description']),axis=1)

# Update the input_df with with empty 'features' columns
input_df[input_df['features'].apply(len) == 0] = missing_features_df

# Check missing Descriptions
missing_desc = input_df[input_df['description'].apply(len) == 0]
print(missing_desc[['description', 'features']])

# # # # # # # # # # # # # # # # # 
# Implement Feature Extraction  # 
# # # # # # # # # # # # # # # # # 

# Text Extraction for 'features' column
features = input_df['features'].to_numpy()
vectorizer_fea = CountVectorizer()
corpus_fea = [" ".join(x) for x in features]
# print(corpus_fea)
X_fea = vectorizer_fea.fit_transform(corpus_fea)
vectorizer_fea.get_feature_names()
X_fea.toarray()

# Text Extraction for 'description' column
descriptions = input_df['description'].to_numpy()
vectorizer_des = CountVectorizer()
corpus_des = descriptions.to_numpy()
# print(corpus_des)
X_des = vectorizer_des.fit_transform(corpus_des)
X_des.toarray()
X_des.get_feature_names()

# # # # # # # # # # # # # # # # # 
#         Extra Comments        # 
# # # # # # # # # # # # # # # # # 


# Which columns seem useful for text feature extraction?
    # Features -> Not much procesing needed.
        # Features is tricky because of nested list of features
    # Discription -> 
        # 1) tokenizing
        # 2) counting
        # 3) normalizing


# print("Input DF is")
# print(input_df['features'].loc[[10]])

# print("Where features don't exist")
# print(input_df[input_df['features'].apply(len) == 0])


# print("Missing Features are")
# print(missing_features_df)


# print("Input DF is")
# print(input_df['features'].loc[[10]])



# left = result[result['features_left'].apply(len) == 0]
# print(left)

# new_df_vectorizer = CountVectorizer()
# corpus = new_df['description'].to_numpy()
# X = new_df_vectorizer.fit_transform(corpus)
# print(new_df_vectorizer.get_feature_names())
# my_list = X.toarray()
# with open('array_output.txt', 'w') as f:
#     for item in my_list:
#         f.write("%s\n" % item)


# vectorizer_des = CountVectorizer()
# corpus_des = descriptions.to_numpy()
# # print(corpus_des)
# X_des = vectorizer_des.fit_transform(corpus_des)
# print(X_des.toarray())
# print(X_des.get_feature_names())

# TOKENIZER IN CountVectorizer expects string or bytes-like object
# [['Doorman, Elevator, Cats Allowed, Dogs Allowed'],['Pre-War, Dogs Allowed, Cats Allowed']] WON'T WORK
# Esentially, we need to convert to ['Doorman, Elevator, Cats Allowed, Dogs Allowed','Pre-War, Dogs Allowed, Cats Allowed']

# This does essentially what is described above
# sample_input = [['Doorman, Elevator, Cats Allowed, Dogs Allowed'],['Pre-War, Dogs Allowed, Cats Allowed']]
# input_corrected = [" ".join(x) for x in sample_input]
# print(input_corrected)

# vectorizer_fea = CountVectorizer()
# corpus_fea = [" ".join(x) for x in features]
# print(corpus_fea)
# X_fea = vectorizer_fea.fit_transform(corpus_fea)
# print(vectorizer_fea.get_feature_names())
# print(X_fea.toarray())