### Milestone 1


## Purpose

### exploring_data.ipynb
The code in this Jupyter Notebook implements the first phase of explotary data analysis. This file outputs histograms for multiple attributes, hourly trends, missing value counts, plots to visualize and count the number of outliers.

### image_feature_extraction.py
Multiple features from images are extracted and plotted. These features include image height, image width, image intensity, saturation, image pixels and image brightness.
### text_feature_extraction_idf.py
The rows with missing features are filled using the corresponding description of the listings. Mean TfIdf values for each listing are inserted into the dataframe for both features and description attributes. In addition, the global tfidf values are plotted for both features and description attributes.
### missing_display_addr.py
The missing display addresses are inferred using the street address attribute.

## Order of operation

The order of operation is irrelevant as all .py and ipynb files are independent of each other.

## Running Python Files

Running exploring_data.ipynb

1. Change directory
`$ cd two-sigma-connect-rental-listing-inquiries/`
2. Start Jupyter Notebook server
`$ jupyter notebook`
3. Click on `exploring_data.ipynb` to open the file
4. Run the file by `Kernal->Restart & Run All`

Running image_feature_extraction.py
`python3 image_feature_extraction.py`

Running text_feature_extraction_idf.py
`python3 text_feature_extraction_idf.py`

Running missing_display_addr.py
`python3 missing_display_addr.py`


## Milestone Progress

* Exploratory data analysis
  - [x] Plot histograms for the following numeric columns: Price, Latitude &
Longitude.
  - [x] Plot hour-wise listing trend and find out the top 5 busiest hours of postings.
  - [x] Visualization to show the proportion of target variable values.
  
* Dealing with missing values, outliers
  - [x] Find out the number of missing values in each variable.
  - [x] Find out the number of outliers in each variable. Plot visualisations to
demonstrate them. Handle outliers.
  - [x] Can we safely drop the missing values? If not, how will you deal with them?
  
* Feature extraction from images and text
  - [x] Extract features from the images and transform it into data that’s ready to be
used in the model for classification.
  - [x] Extract features from the text data and transform it into data that’s ready to be
used in the model for classification.

## Libraries Used
- numpy
- pandas
- seaborn
- scipy
- matplotlib
- sklearn
- pillow
- nltk

