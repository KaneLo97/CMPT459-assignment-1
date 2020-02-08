import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys 
import matplotlib.pyplot as plt


input_df = pd.read_json('train.json.zip')

new_df = input_df[['manager_id', 'listing_id']]

agg_count = new_df.groupby('manager_id').count().sort_values(by='listing_id',ascending=False)
print(agg_count)

# plt.bar(x=agg_count['manager_id'], y=agg_count['listing_id'])
# plt.show()

# pd.plot(x=agg_count['manager_id'], y=agg_count['listing_id'])
agg_count.to_csv('counts.csv')