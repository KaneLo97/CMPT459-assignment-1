import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys 
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import re


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('popular')

# Settings for Terminal 
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# np.set_printoptions(threshold=sys.maxsize)

my_stop_words = list(stopwords.words('english')) 


# class LemmaTokenizer:
#      def __init__(self):
#          self.wnl = WordNetLemmatizer()
#      def __call__(self, doc):
#          return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



input_df = pd.read_json('train.json.zip')

# descriptions = input_df['description']
# features = input_df[['features', 'description', 'listing_id']]
descriptions = input_df
features = input_df
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
# Remove HTML tags such as br   # 
# # # # # # # # # # # # # # # # # 
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

# Apply the function to remove tags
input_df['description'] = input_df.apply(lambda row: remove_tags(row['description']),axis=1)

# # # # # # # # # # # # # # # # # 
# Implement Feature Extraction  # 
# # # # # # # # # # # # # # # # # 

# Text Extraction for 'features' column
features = input_df['features'].to_numpy()
vectorizer_fea = CountVectorizer(stop_words=my_stop_words)
corpus_fea = [" ".join(x) for x in features]
# print(corpus_fea)
X_fea_counts = vectorizer_fea.fit_transform(corpus_fea)
# features_list = vectorizer_fea.get_feature_names()
# print(features_list)
# print("Length of features list is: ", len(features_list))
# features_list = X_fea.toarray()

# To TfIDF
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X_fea_counts)

# print(tfidf)
# print("DONE")
# print(tfidf[0])
# print(tfidf.shape[0])
# print(tfidf[0, :].toarray())
# print(tfidf[49351].toarray()[0])
# print("DONE")
# print(tfidf[126].toarray())

# Add the Mean Tf/Idf for each row in the database for Features
num_of_rows = tfidf.shape[0]
mean_val_list = []
for row in range(num_of_rows):
    elem_list = tfidf[row].toarray()[0]
    # print("Row #: ", row)
    # print("Len of this arr is ", len(elem_list))
    row_vals = []
    mean_val = 0
    for elem in elem_list:
        if (float(elem) != float(0.0)):
            # print(elem)
            row_vals.append(float(elem))
    if len(row_vals) > 0:
        mean_val = statistics.mean(row_vals)
        print("Mean is ", mean_val)
        mean_val_list.append(mean_val)
    else:
        # print("Mean is ", 0.0)
        mean_val_list.append(0.0)

input_df['mean_feature_tdidf'] = mean_val_list
# print(input_df)


# Text Extraction for 'description' column
descriptions = input_df['description'].to_numpy()
vectorizer_des = CountVectorizer(stop_words=my_stop_words)
corpus_des = descriptions
X_des_counts = vectorizer_des.fit_transform(corpus_des)
# To TfIDF
des_transformer = TfidfTransformer(smooth_idf=False)
des_tfidf = des_transformer.fit_transform(X_des_counts)
# print(vectorizer_des.get_feature_names())

# # Add the Mean Tf/Idf for each row in the database for Description
num_of_rows = des_tfidf.shape[0]
mean_val_list = []
for row in range(num_of_rows):
    elem_list = des_tfidf[row].toarray()[0]
    # print("Row #: ", row)
    # print("Len of this arr is ", len(elem_list))
    row_vals = []
    mean_val = 0
    for elem in elem_list:
        if (float(elem) != float(0.0)):
            # print(elem)
            row_vals.append(float(elem))
    if len(row_vals) > 0:
        mean_val = statistics.mean(row_vals)
        print("Mean is ", mean_val)
        mean_val_list.append(mean_val)
    else:
        # print("Mean is ", 0.0)
        mean_val_list.append(0.0)

input_df['mean_des_tdidf'] = mean_val_list
# print(input_df)

# # # # # # # # # # # # # # # # # 
#  Feature TfIdf Graph          # 
# # # # # # # # # # # # # # # # # 

# Global IDF for Features (ascending)
df_idf = pd.DataFrame(transformer.idf_, index=vectorizer_fea.get_feature_names(),columns=["idf_weights"]).reset_index()

# sort ascending
sorted_df = df_idf.sort_values(by=['idf_weights'])
print(sorted_df)
# sorted_df.to_csv("idf.csv")
sns.barplot(sorted_df['idf_weights'],sorted_df['index'][0:10,])
plt.xlabel("IDF Weight", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature vs. IDF Weight")
# plt.show()
plt.savefig('feature_tfidf_asc.png')


# # # Global IDF for Features (descending)
sorted_df_desc = df_idf.sort_values(by=['idf_weights'], ascending=False)
print(sorted_df_desc)
sns.barplot(sorted_df_desc['idf_weights'],sorted_df_desc['index'][0:10,])
plt.xlabel("IDF Weight", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature vs. IDF Weight")
# plt.show()
plt.savefig('feature_tfidf_dsc.png')


# # # # # # # # # # # # # # # # # 
#  Description TfIdf Graph      # 
# # # # # # # # # # # # # # # # # 

df_idf_description = pd.DataFrame(des_transformer.idf_, index=vectorizer_des.get_feature_names(),columns=["idf_weights"]).reset_index()
# sort ascending
sorted_df_description = df_idf_description.sort_values(by=['idf_weights'])
print(sorted_df_description)
sns.barplot(sorted_df_description['idf_weights'],sorted_df_description['index'][0:10,])
plt.xlabel("IDF Weight", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature vs. IDF Weight")
# plt.show()
plt.savefig('desc_tfidf_asc.png')


# Global IDF for Features (descending)
sorted_df_description_desc = df_idf_description.sort_values(by=['idf_weights'], ascending=False)
print(sorted_df_description_desc)
sns.barplot(sorted_df_description_desc['idf_weights'],sorted_df_description_desc['index'][0:10,])
plt.xlabel("IDF Weight", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature vs. IDF Weight")
plt.show()
# plt.savefig('desc_tfidf_dsc.png')



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


# with open('array_output.txt', 'w') as f:
#     for item in my_list:
#         f.write("%s\n" % item)



# TOKENIZER IN CountVectorizer expects string or bytes-like object
# [['Doorman, Elevator, Cats Allowed, Dogs Allowed'],['Pre-War, Dogs Allowed, Cats Allowed']] WON'T WORK
# Esentially, we need to convert to ['Doorman, Elevator, Cats Allowed, Dogs Allowed','Pre-War, Dogs Allowed, Cats Allowed']