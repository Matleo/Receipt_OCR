# -*- coding: utf-8 -*-

from glob import glob
import cv2
import pickle
import numpy as np
import sys
import json
import os

src_dir = os.path.dirname(os.path.realpath(__file__))

#Utilities:
def find_alphanum_label(int_labels,customer):
    labels = []
    filename = src_dir +"/../assets/"+customer+"/Outputs/TrainingData/alphanum_label_dict.json"
    with open(filename, "r", encoding="utf8") as infile:
        dict = json.load(infile)
    for int_label in int_labels:
        for key, value in dict.items():
            if value == int_label:
                labels.append(key)
    return labels


#function to subset data (for example used if not wanting to use full dataset of spaces)
def slice_data(dataToSlice, newDataName, by):
    with open(dataToSlice, "rb") as input_file:
        data = pickle.load(input_file)
        length = int(len(data["x"]) / by)
        data["x"] = data["x"][:length]
        data["y"] = data["y"][:length]
        data["file_names"] = data["file_names"][:length]
    with open(newDataName, 'wb') as outfile:
        pickle.dump(data, outfile)


#actual preprocessing:
def preprocess_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray_image, dsize=(14, 34), interpolation=cv2.INTER_NEAREST)
    res = res / 255 #scaling
    return res

def open_img(file):
    with open(file, "rb") as infile:
        bytes = bytearray(infile.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return img

def read_images(imageFolder, charList):
    label_dict_filename = src_dir +"/../assets/"+customer+"/Outputs/TrainingData/alphanum_label_dict.json"
    with open(label_dict_filename, "r", encoding="utf8") as infile:
        label_dict = json.load(infile)

    resized_images = []
    labels = []
    file_names = []
    for i in range(len(charList)):
        char = charList[i]
        charCode = char[0:3]
        print("processing folder: " + imageFolder + char)
        files = glob(imageFolder+char+"/*.png")
        for file in files:
            file = file.replace("\\","/")
            img = open_img(file)
            res = preprocess_image(img)
            resized_images.append(res)

            label = label_dict[charCode]
            labels.append(label)
            file_names.append(file)
        # maybe do standardization for features: (x - x_mean) / x_std
    return resized_images, labels, file_names

def save_data(features, labels, file_names, imageFolderNumber, customer):
    data = {"x": features, "y": labels, "file_names": file_names}
    data_filename = src_dir+'/../assets/'+customer+'/Outputs/TrainingData/data_' + imageFolderNumber + '.pickle'
    with open(data_filename, 'wb') as outfile:
        pickle.dump(data, outfile)
    print("pickled dataset to: " + data_filename+"\n")

def main():
    print("Preprocessing for "+customer+" has started")
    fontSizes = glob(src_dir+"/../assets/"+customer+"/Characters/*/")
    for imageFolder in fontSizes:
        imageFolder = imageFolder.replace("\\","/")
        charList = glob(imageFolder + "*/")
        for i in range(len(charList)):
            charList[i] = charList[i].replace("\\", "/").replace(imageFolder, "")[:-1] #folder
        resized_images, labels, file_names = read_images(imageFolder, charList)
        imageFolderNumber = imageFolder.split("/")[-2]
        save_data(resized_images, labels, file_names, imageFolderNumber, customer)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        customer = sys.argv[1]
    else:
        customer = "MüllerNew"
    main()
    #create_alphanum_label_dict()
    #slice_data("../assets/Müller/data_space.pickle","../assets/Müller/data_space_quarter.pickle",4)
