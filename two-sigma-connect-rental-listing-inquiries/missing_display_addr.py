import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys 

input_df = pd.read_json('train.json.zip')

new_df = input_df[['display_address', 'street_address']]

no_display_addr = new_df[new_df['display_address'].apply(len) == 0]
print("BEFORE REPLACING VALS")
print(no_display_addr)


no_display_addr['display_address'] = no_display_addr['street_address']
print("AFTER REPLACING VALS")
print(no_display_addr)

# To check if the code works
# new_no_display_addr = no_display_addr[no_display_addr['display_address'].apply(len) == 0]
# print(new_no_display_addr)