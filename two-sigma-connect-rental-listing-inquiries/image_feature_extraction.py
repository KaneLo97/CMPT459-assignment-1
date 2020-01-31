import numpy as np
import pandas as pd
import os
import sys
import re
import operator
from zipfile import ZipFile
from subprocess import check_output
from PIL import Image

def exploreFiles():
    
    root_dir = os.getcwd() #get the current directory
    folder = []

    #read all the files
    for root, sub_dirs, _ in os.walk(root_dir):
        for dir in sub_dirs:
            dir_path = os.path.join(root, dir)
            di = os.path.basename(dir_path)
            if(di.isnumeric()):
                entries = os.listdir(dir_path)
                #print (len(entries))
                for entry in entries:
                    folder.append(di)
    #print (folder)           
    return folder

def exploreImages():
    root_dir = os.getcwd() #get the current directory
    images_list = []
    complete_images_list = []

    #read all the files
    for root, sub_dirs, _ in os.walk(root_dir):
        for dir in sub_dirs:
            dir_path = os.path.join(root, dir)
            dir_basename = os.path.basename(dir_path)
            if(dir_basename.isnumeric()):
                entries = os.listdir(dir_path)
                for entry in entries: 
                    images_list.append(entry) #add all the images belonging to the listing_id to the images_list
                if len(images_list) != 0:
                   complete_images_list.append((dir_basename,images_list)) #an array of tuples(folder name, folder images)
                images_list = []
               
    return complete_images_list


def process_image(path):

    path = './images_sample/'+path[0:7]+'/'+path
    img = np.array(Image.open(path))
    #get dims
    width = img.shape[1]
    height = img.shape[0]

    #flatten the image
    img = img.transpose(2,0,1).reshape(3,-1)

    #brightness is simple, assign 1 if zero to avoid divide
    bright = np.amax(img,axis=0)
    bright[bright==0] = 1

    #hue, same, assign 1 if zero, not working atm due to arccos
    denom = np.sqrt((img[0]-img[1])**2-(img[0]-img[2])*(img[1]-img[2]))
    denom[denom==0] = 1

    #saturation
    sat = (bright - np.amin(img,axis=0))/bright

    #return mean values
    return width,height,np.mean(bright),np.mean(sat)

    #print(img)

def process_row(row, image_files):
    result = []
    for image in image_files:
        if image[0] == str(row.listing_id):
            # print (image[0])
            # print (image)
            result = np.array([process_image(x) for x in image[1]])
    result = np.mean(result,axis=0)
    row['image_width'] = result[0]
    row['image_height'] = result[1]
    row['image_brightness'] = result[2]
    row['image_saturation'] = result[3]
    return row
    
'''  
def process_row(row):
    images = check_output(["ls", "./images_sample/"+str(row.listing_id)]).decode("utf8").strip().split('\n')
    res = np.array([process_image(x) for x in images])
    #print(images)
    res = np.mean(res,axis=0)
    row['image_width'] = res[0]
    row['image_height'] = res[1]
    row['image_brightness'] = res[2]
    row['image_saturation'] = res[3]
    return row
'''


def main():
# 
#     with ZipFile('images_sample.zip', 'r') as zipObj:
#         # Extract all the contents of zip file in current directory
#         zipObj.extractall()
#         
    
    df = pd.read_json('train.json.zip')
    folder_files = exploreFiles()
    image_files = exploreImages()
    df = df[df.listing_id.isin(folder_files)]
    df = df.filter(['photos', 'listing_id'])
    df['num_photos'] = df.apply(lambda x: len(x['photos']), axis = 1)
    #print(df)
    df = df[df['num_photos'] != 0]
    

    #df = df.apply(lambda row: process_row(row),axis=1)
    df = df.apply(lambda row: process_row(row, image_files),axis=1)
    print (df)
            

if __name__ == "__main__":
    main()
