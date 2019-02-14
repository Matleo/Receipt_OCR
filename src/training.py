import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import random
from preprocessing import find_alphanum_label
import json
import io
import os
from glob import glob
import sys

batch_size = 64
num_classes = 0 #will be updated in main
epochs = 50
input_shape = (34, 14, 1)
src_dir = os.path.dirname(os.path.realpath(__file__))


def load_data(dataset_paths):
    data = {"x": [], "y": [], "file_names": []}
    for dataset_path in dataset_paths:
        with open(dataset_path, "rb") as input_file:
            tmp_data = pickle.load(input_file)
            print("Opened data from: " + dataset_path)
            data["x"].extend(tmp_data["x"])
            data["y"].extend(tmp_data["y"])
            data["file_names"].extend(tmp_data["file_names"])

    features = np.asarray(data["x"]).astype("float32")
    labels = np.asarray(data["y"]).astype("int32")
    file_names = np.asarray(data["file_names"])

    features = features.reshape(features.shape[0], input_shape[0], input_shape[1], input_shape[2])
    labels = keras.utils.to_categorical(labels, num_classes)

    return features, labels, file_names

def do_cross_validation(features, labels, file_names, k, customer_name):
    random.seed(42)
    shuffled_indice = list(range(len(features)))
    random.shuffle(shuffled_indice)

    end = -1
    test_size = int(features.shape[0] / k)
    Y_pred_full = []
    Y_true_full = []
    filenames_full = []
    class_labels = []
    char_accs = []
    accs = []

    #k-fold:
    for i in range(k):
        print("\n----------------"+str(i+1)+"-fold----------------")
        start = end + 1
        end = start + test_size
        if i == k-1: #last step
            end = features.shape[0] - 1

        test_indice = shuffled_indice[start:end]
        train_indice = [shuffled_indice[i] for i in range(len(shuffled_indice)) if i not in range(start, end)]
        train = {"x": features[train_indice], "y": labels[train_indice], "file": file_names[train_indice]}
        test = {"x": features[test_indice], "y": labels[test_indice], "file": file_names[test_indice]}

        Y_pred, Y_true, test_filenames, class_labels_tmp, acc, char_acc = start_training(train, test, False, customer_name)
        Y_pred_full.extend(Y_pred)
        Y_true_full.extend(Y_true)
        filenames_full.extend((test_filenames))
        class_labels = class_labels_tmp
        char_accs.append(char_acc)
        accs.append(acc)

    save_wrong_classifications(Y_pred_full, Y_true_full, filenames_full, class_labels, np.mean(accs), np.mean(char_accs), customer_name)

def start_training(train, test, full_train, customer_name):
    print("Train-set is of size: " + str(len(train["x"])))
    print("Test-set is of size: " + str(len(test["x"])) + "\n")
    model = create_CNN()

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(train["x"])
    model.fit_generator(datagen.flow(train["x"], train["y"], batch_size=batch_size),
                            epochs=epochs, verbose=2,
                            steps_per_epoch=train["x"].shape[0]/batch_size,
                            validation_data=(test["x"],test["y"]))

    #save model:
    if full_train:
        model.save(src_dir+"/../assets/"+customer_name+"/Outputs/model.h5")

        # frozen_model:
        sess = K.get_session()
        export_dir_frozenModel = src_dir+"/../assets/"+customer_name+"/Outputs"
        #create folder
        if not os.path.exists(export_dir_frozenModel):
            os.makedirs(export_dir_frozenModel)
        #remove file if exists
        if os.path.isfile(export_dir_frozenModel+"/frozenModel.pb"):
            os.remove(export_dir_frozenModel+"/frozenModel.pb")
        output_node_names = [node.op.name for node in model.outputs] #dense_2/Softmax
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            sess.graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(export_dir_frozenModel+"/frozenModel.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())

    return evaluate_model(model, test, full_train, customer_name)


def evaluate_model(model, test, full_train, customer_name):
    final_loss, final_acc = model.evaluate(test["x"], test["y"], verbose=0)
    print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))

    # Predict the values from the validation dataset
    Y_pred_onehot = model.predict(test["x"])
    # Convert predictions classes to one hot vectors
    Y_pred = np.argmax(Y_pred_onehot, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(test["y"], axis=1)


    #compute character_accuracy (whitout 0-O etc)
    class_labels_full = find_alphanum_label(list(range(num_classes)), customer_name)
    char_acc = char_accuracy(Y_pred, Y_true, class_labels_full)

    #save on which test data the classifier failed:
    if full_train:
        save_wrong_classifications(Y_pred, Y_true, test["file"], class_labels_full, final_acc, char_acc, customer_name)

    return Y_pred, Y_true, test["file"], class_labels_full, final_acc, char_acc

def create_CNN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2)))  # mal auslassen?
    model.add(Dropout(0.20))  # 20
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # 25
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))  # 25
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))  # 25
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

def char_accuracy(y_pred, y_true, class_labels):
    correct_classifications = 0
    for i in range(len(y_true)):
        if pred_is_correct(y_true[i], y_pred[i], class_labels):
            correct_classifications += 1
    return float(correct_classifications/len(y_true))


def save_wrong_classifications(y_pred, y_true, file_names, class_labels, accuracy, char_accuracy, customer_name):
    error_report = {}
    error_report["accuracy"] = accuracy
    error_report["char_accuracy"] = char_accuracy
    error_report["test_size"] = len(y_true)
    error_report["epochs"] = epochs
    mistakes = []
    for i in range(len(y_pred)):
        if not pred_is_correct(y_pred[i], y_true[i], class_labels):
            y_pred_char = label_dict[str(y_pred[i])]
            y_true_char = label_dict[str(y_true[i])]
            mistake = {"label": y_true_char, "pred": y_pred_char, "file": file_names[i]}
            mistakes.append(mistake)
    error_report["mistakes"] = mistakes
    json_path = src_dir+"/../assets/"+customer_name+"/Outputs/error_report.json"
    with io.open(json_path, "w", encoding="utf8") as outfile:
        json.dump(error_report, outfile, ensure_ascii=False, indent=1)
    print("Saved error report for " + str(len(mistakes))+ " wrong classifications to: "+json_path+". (char_acc: " + str(char_accuracy) + ")")

#deprecated and unused. for old data format of mulller
def pred_is_correct_muller(prediction, true_class, class_labels):
    #find relevant indice
    for i in range(len(class_labels)):
        if class_labels[i] == "O-upper":
            index_O = i
        if class_labels[i] == "o-lower":
            index_o = i
        if class_labels[i] == "pct-44":
            index_komma = i
        if class_labels[i] == "pct-46":
            index_punkt = i
        if class_labels[i] == "l-lower":
            index_l = i
        if class_labels[i] == "I-upper":
            index_I = i
        if class_labels[i] == "1":
            index_1 = i
        if class_labels[i] == "w-lower":
            index_w = i
        if class_labels[i] == "W-upper":
            index_W = i
        if class_labels[i] == "v-lower":
            index_v = i
        if class_labels[i] == "V-upper":
            index_V = i
        if class_labels[i] == "Z-upper":
            index_Z = i
        if class_labels[i] == "z-lower":
            index_z = i
        if class_labels[i] == "s-lower":
            index_s = i
        if class_labels[i] == "S-upper":
            index_S = i
        if class_labels[i] == "c-lower":
            index_c = i
        if class_labels[i] == "C-upper":
            index_C = i
        if class_labels[i] == "u-lower":
            index_u = i
        if class_labels[i] == "U-upper":
            index_U = i
        if class_labels[i] == "K-upper":
            index_K = i
        if class_labels[i] == "k-lower":
            index_k = i
        if class_labels[i] == "Ü-upper":
            index_Ü = i
        if class_labels[i] == "ü-lower":
            index_ü = i
        if class_labels[i] == "x-lower":
            index_x = i
        if class_labels[i] == "X-upper":
            index_X = i


    if prediction == true_class:
        return True
    elif (prediction in [index_O, index_o]) and (true_class in [index_O, index_o]):
        return True
    # else if one says , and other says .
    elif (prediction in [index_punkt, index_komma]) and (true_class in [index_punkt, index_komma]):
        return True
    # misprediction I vs l vs 1
    elif (prediction in [index_I, index_l, index_1]) and (true_class in [index_I, index_l, index_1]):
        return True
    elif (prediction in [index_w, index_W]) and (true_class in [index_w, index_W]):
        return True
    elif (prediction in [index_z, index_Z]) and (true_class in [index_z, index_Z]):
        return True
    elif (prediction in [index_v, index_V]) and (true_class in [index_v, index_V]):
        return True
    elif (prediction in [index_s, index_S]) and (true_class in [index_s, index_S]):
        return True
    elif (prediction in [index_c, index_C]) and (true_class in [index_c, index_C]):
        return True
    elif (prediction in [index_u, index_U]) and (true_class in [index_u, index_U]):
        return True
    elif (prediction in [index_ü, index_Ü]) and (true_class in [index_ü, index_Ü]):
        return True
    elif (prediction in [index_x, index_X]) and (true_class in [index_x, index_X]):
        return True
    elif (prediction in [index_k, index_K]) and (true_class in [index_k, index_K]):
        return True
    elif (prediction in [index_Ü, index_O]) and (true_class in [index_Ü, index_O]):
        return True
    else:
        return False
#compare characters with substitutions (super ugly, please dont look at this :D)
def pred_is_correct(prediction, true_class, class_labels):
    index_0 = None  # maybe no "048-0" exists
    index_O = None
    index_o = None
    index_komma = None
    index_punkt = None
    index_I = None
    index_1 = None
    index_w = None
    index_W = None
    index_v = None
    index_V = None
    index_Z = None
    index_z = None
    index_s = None
    index_S = None
    index_c = None
    index_C = None
    index_u = None
    index_U = None
    index_K = None
    index_k = None
    #find relevant indice
    for i in range(len(class_labels)):
        if class_labels[i] == "048":
            index_0 = i
        if class_labels[i] == "079":
            index_O = i
        if class_labels[i] == "111":
            index_o = i
        if class_labels[i] == "044":
            index_komma = i
        if class_labels[i] == "046":
            index_punkt = i
        if class_labels[i] == "108":
            index_l = i
        if class_labels[i] == "073":
            index_I = i
        if class_labels[i] == "049":
            index_1 = i
        if class_labels[i] == "119":
            index_w = i
        if class_labels[i] == "087":
            index_W = i
        if class_labels[i] == "118":
            index_v = i
        if class_labels[i] == "086":
            index_V = i
        if class_labels[i] == "090":
            index_Z = i
        if class_labels[i] == "122":
            index_z = i
        if class_labels[i] == "115":
            index_s = i
        if class_labels[i] == "083":
            index_S = i
        if class_labels[i] == "099":
            index_c = i
        if class_labels[i] == "067":
            index_C = i
        if class_labels[i] == "117":
            index_u = i
        if class_labels[i] == "085":
            index_U = i
        if class_labels[i] == "075":
            index_K = i
        if class_labels[i] == "107":
            index_k = i

    if prediction == true_class:
        return True
    elif (prediction in [index_O, index_o, index_0]) and (true_class in [index_O, index_o, index_0]):
        return True
    # else if one says , and other says .
    elif (prediction in [index_punkt, index_komma]) and (true_class in [index_punkt, index_komma]):
        return True
    # misprediction I vs l vs 1
    elif (prediction in [index_I, index_l, index_1]) and (true_class in [index_I, index_l, index_1]):
        return True
    elif (prediction in [index_w, index_W]) and (true_class in [index_w, index_W]):
        return True
    elif (prediction in [index_z, index_Z]) and (true_class in [index_z, index_Z]):
        return True
    elif (prediction in [index_v, index_V]) and (true_class in [index_v, index_V]):
        return True
    elif (prediction in [index_s, index_S]) and (true_class in [index_s, index_S]):
        return True
    elif (prediction in [index_c, index_C]) and (true_class in [index_c, index_C]):
        return True
    elif (prediction in [index_u, index_U]) and (true_class in [index_u, index_U]):
        return True
    elif (prediction in [index_k, index_K]) and (true_class in [index_k, index_K]):
        return True
    else:
        return False


def main(customer_name, train_full):
    dataset_paths = glob(src_dir+"/../assets/"+customer_name+"/Outputs/TrainingData/*.pickle")
    for i in range(len(dataset_paths)):
        dataset_paths[i] = dataset_paths[i].replace("\\","/")
    print("Loading dataset:")
    features, labels, file_names = load_data(dataset_paths)

    print("Starting training now:")
    if not train_full:
        k = 10
        do_cross_validation(features, labels, file_names, k, customer_name)
    else:
        train = {"x": features, "y": labels, "file": file_names}
        start_training(train, train, True, customer_name)

if __name__ == "__main__":
    customer_name = "DennsNew"
    train_full = True
    if len(sys.argv) == 2:
        customer_name = sys.argv[1]
    if len(sys.argv) == 3:
        customer_name = sys.argv[1]
        if sys.argv[2] in ["crossvalidate","cv"]:
            train_full = False

    # read how many classes are defined in alphanum_label_dict.json
    filename = src_dir +"/../assets/"+customer_name+"/Outputs/char_label_dict.json"
    with open(filename, "r", encoding="utf8") as infile:
        label_dict = json.load(infile)
    num_classes = len(label_dict)

    main(customer_name, train_full = train_full)
