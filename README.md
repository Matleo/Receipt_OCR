# OCR-NN Documentation
## Introduction
This Project was developed by Matthias Leopold for the [RFND AG](http://rfnd.com/).
This is a convolutional neural network which is used in the self-service automat and is resoponsible to classify the characters on the receipt, based on the layout analysis by Matthias Heubi.

## Project structure
In the following i will describe the meaning and content of each directory
### assets
This folder holds the root folders for each customer. The content of each customer Folder needs to follow this naming convention:
* assets
	* *CustomerA*
		* **Characters**
			* *FontSizeX*
				* 035-hash_sign
				* 037-percent_sign
				* ...
			* *FontSizeY*
				* 035-hash_sign
				* ...
		* **Scans**
			* tx_1.jpg
			* tx_1.jpg.json
			* ...
	* *CustomerB*
		* ... 

**Characters**: This folder contains the learning data images (one character per image). In there is a folder for each font size that was extracted from the customer Receipts (min. one font size folder). Each font size folder contains folders for the actual character images, named following the convention of \<unicode\>-\<human readable char representation\>. For example, the folder *CustomerA/Characters/FontSizeX/035-hash_sign* contains all images of hash signs in font size *FontSizeX* that were obtained from *CustomerA*.

**Scans**: This folder contains the full receipt images and its corresponding layout json files. Note that the json file needs to have the exact same name as the jpg image, but with an extra ".json" extension.

The training process will also create other artefacts inside the Customer Folder (next to the **Characters** folder) that are not initially present, like the preprocessed data and the model.

### src
This is the source folder and contains the actual python source code. Following, i will shortly descripe its content:
* **prediction_service.py:** The callable service, that will fill out the empty layout files in the **Scans** folder.
* **preprocessing.py:** The Script that creates the actual learning data from the Characters images. It resizes the image to a standard size, converts the colors to greyscale and appends the corresponding label to the data. It creates the *data_\<FontSize\>.pickle* files inside the Customer folder.
* **training.py:** The script that takes the preprocessed data and trains the actual CNN. It creates the *frozenModel* folder, the *error_report.json* file and the *model.h5* file inside the Customer Folder.

## Usage
### Prerequisites
You will need to have Python>=3.7 installed on your device. You can install all pip dependencies using the *pip-requirements.txt* provided in this project:

	pip install -r pip-requirements.txt

### Preprocessing
If you have set up the folder structe for a new Customer correctly (as descriped in Project Structure), you can start preprocessing. The only command line argument that you have to pass is the name of the Customer folder:

	python preprocessing.py CustomerA
    
### Training
After the Preprocessing is done, there should be *data_\<FontSize\>.pickle* files inside the Customer folder, according to the FontSize folders inside the Characters folder. Now you can start the actual training, again passing the Customer folder name:

	python training.py CustomerA
    
The training will output three kind of artifacts:
* **error_report.json**: Informations about character accuracy and miss-classifications
* **model.h5**: The trained model with its architecture and parameters, to reuse in python
* **frozenModel/model.pb**: Trained model, to reuse in c#

The python command above will take the full training data and output a model that can be used for inference. Before you want to do this, you will most likely want to test how good the network performs on the data. Therefore you can pass `crossvalidate` as a second command line argument to indicate that you do not want to train on the full data, but rather perform a 10-fold cross validation. The error_report accuracy will be an average and the missclassifications will be saved over all folds:

	python training.py CustomerA crossvalidate 
    
### Prediction
After the model was trained, you can start using it to fill out the layouts.
#### Python
For the `prediction_service.py` you will have to specify for which Customer to fill out the layout files. Note that the "empty" layout files from the layout analysis (which only define the boundingboxes of the characters) need to be put into the Customer folder before running the prediction. These files will then be filled out with the predicted character values:

	python prediction_service.py CustomerA

By default, the `model.h5` from the Customer folder is used for prediction.