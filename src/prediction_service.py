from keras.models import load_model
import numpy as np
import json
from preprocessing import open_img, preprocess_image
from glob import glob
import io
import time
import sys

class predictor():
    def __init__(self, modelpath, label_dic_path):
        self.model = load_model(modelpath)
        print("Loaded Keras model from: " + modelpath)
        self.dataShape = [34,14]

        dic_filename = label_dic_path
        with open(dic_filename, "r", encoding="utf8") as infile:
            self.dict = json.load(infile)

    def predict(self, data):
        data = np.asarray(data).astype("float32")
        data = data.reshape(data.shape[0], self.dataShape[0], self.dataShape[1], 1)
        predictionsOnehot = self.model.predict(data)
        confidences = []
        predictions = []
        for pred in predictionsOnehot:
            int_label = np.argmax(pred)
            prediction = self.find_char_label(int_label)
            confidence = float(pred[int_label])
            predictions.append(prediction)
            confidences.append(confidence)
        return predictions, confidences

    def load_data(self, filename):
        img = open_img(filename)
        data = preprocess_image(img)
        return data

    def crop_preprocess_image(self, full_img , bbox):
        bbox_array = bbox.split(" ")
        links = int(bbox_array[0])
        oben = int(bbox_array[1])
        rechts = int(bbox_array[2])
        unten = int(bbox_array[3])
        crop_img = full_img[oben:unten, links:rechts]
        data = preprocess_image(crop_img)
        return data

    def find_char_label(self, int_label):
        return self.dict[str(int_label)]

    def predict_full_layout(self, folder, image):
        filled_layout = {"Lines": []}
        json_filename = folder+"/"+image+".json"
        with open(json_filename, "r", encoding="utf8") as infile:
            layout = json.load(infile)
        layout_lines = layout["Lines"]

        full_img = open_img(folder+"/"+image)
        #step 1: extract data from image
        all_data = []
        index = 0
        for line in layout_lines:
            filled_line = {"Words": []}
            for word in line["Words"]:
                if "IsPhantom" in word:
                    filled_word = {"IsPhantom": word["IsPhantom"], "BoundingBox": word["BoundingBox"], "Characters": []}
                else:
                    filled_word = {"BoundingBox": word["BoundingBox"], "Characters": []}

                for char in word["Characters"]:
                    bbox = char["BoundingBox"]
                    #actual prediction here:
                    data = self.crop_preprocess_image(full_img, bbox)
                    #filled_char = {"BoundingBox": bbox, "Text": prediction, "Confidence": confidence}
                    filled_char = {"BoundingBox": bbox, "Index": index}
                    index += 1
                    all_data.append(data)
                    filled_word["Characters"].append(filled_char)
                filled_line["Words"].append(filled_word)
            filled_layout["Lines"].append(filled_line)

        #step 2: batch-classify all cropped images:
        predictions, confidences = self.predict(all_data)
        for line in filled_layout["Lines"]:
            for word in line["Words"]:
                for char in word["Characters"]:
                    i = char["Index"]
                    char["Text"] = predictions[i]
                    char["Confidence"] = confidences[i]
                    del char["Index"] #drop Index field from dict

        return filled_layout

    def save_layout(self, layout, json_filename):
        with io.open(json_filename, "w", encoding="utf8") as outfile:
            json.dump(layout, outfile, ensure_ascii=False, indent=1)

def main(customer, model_name):
    picFolder = "../assets/"+customer+"/Scans"
    model_path = "../assets/"+customer+"/Outputs/"+model_name+".h5"
    label_dic_path = "../assets/"+customer+"/Outputs/char_label_dict.json"

    pred = predictor(model_path, label_dic_path)
    images = glob(picFolder+"/*.jpg")
    start = time.time()
    for i in range(len(images)):
        img = images[i]
        img = img.replace("\\","/")
        img = img.replace(picFolder,"")[1:]
        startSingle = time.time()
        layout = pred.predict_full_layout(picFolder, img)
        endSingle = time.time()
        print("(" + str(i+1) + "|" + str(len(images)) + "): " + img + " | "+str(round(endSingle - startSingle , 2)) + " seconds")

        json_filename = picFolder+"/"+img+".json"
        pred.save_layout(layout, json_filename)
    end = time.time()
    print("Prediction took: " + str(round(end - start , 2)) + " seconds")

if __name__ == "__main__":
    customer = "Test"
    model_name = "model"
    if len(sys.argv) == 2:
        customer = sys.argv[1]
    if len(sys.argv) == 3:
        customer = sys.argv[1]
        model_name = sys.argv[2]
    main(customer, model_name)

    '''
    pred = predictor("../assets/Denns/model.h5", "../assets/Denns/char_label_dict.json")
    data = pred.load_data("../assets/Müller/Characters/space/032-space/tx_2_2-296-12.png")
    prediction = pred.predict([data])
    '''