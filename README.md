# FEDn-ImageSegmentation-tutorial

This is an example tutorial for how to set up and train a Image segmentation model in a federated setting using FEDn. 
Dataset using for this example: Brats2020
Model: U-net implemented in keras.

## Pre configuration
*Requirements: Python with numpy*

- Clone this repo.
- Create a data folder inside of this repo:
```
mkdir data
```
- Insert the dataset: Brats_2020_slices in the data folder.
- Create data partitions:
```
python create_data_partitions.py [NR_OF_CLIENTS] [TRAINING_SUBJECTS_PER_CLIENTS] [VALIDATION_SUBJECTS_PER_CLIENTS]
```
Constrains: \[NR_OF_CLIENTS] **x** **(** \[TRAINING_SUBJECTS_PER_CLIENTS] **+** \[VALIDATION_SUBJECTS_PER_CLIENTS] **)** **<=** 369


## Setting up pseudo deployment

- clone the FEDn repo:
```
git clone -b develop https://github.com/scaleoutsystems/fedn.git
```
For a detailed description visit: https://scaleoutsystems.github.io/fedn/deployment.html
To set up the Base, Reducer and Combiner:
### Base Container
Open a new terminal and navigate to the fedn repository folder and type:
```
docker network create fedn_default
docker-compose -f config/base-services.yaml up
```
### Reducer Container
Open a new terminal and navigate to the fedn repository folder and type:
Copy the file “config/settings-reducer.yaml.template” to “config/settings-reducer.yaml”.
```
docker-compose -f config/reducer-dev.yaml -f config/private-network.yaml up 
```
### Combiner Container
Open a new terminal and navigate to the fedn repository folder and type:
Copy ‘config/settings.yaml.template’ to ‘config/settings-combiner.yaml’.
```
docker-compose -f config/combiner-dev.yaml -f config/private-network.yaml up 
```
### Configurate the Federation
- Open the Reducer UI: https://localhost:8090
- Choose keras-helper and Package: package.tar.gz (from the FEDn-ImageSegmentation-tutorial repository).
- Set the initial model: initial_weights.npz (from the FEDn-ImageSegmentation-tutorial repository)
- Go to Network and download client config to the FEDn-ImageSegmentation-tutorial repository folder.

### Client Container
Open a new terminal and navigate to the FEDn-ImageSegmentation-tutorial repository folder and type:

```
docker-compose -f docker-compose.yaml -f private-network.yaml up 
```
### Federated Training
- Start the training from the Reducer UI -> Control

Here can you set: 
- Number of rounds, 
- Round timeout - the maximum amount of time the aggregators will wait for a model update,
- Validate - Choose if you want to run a validation call on each client on every global model.
### Global Validation
- To see global model validation go to Reducer UI -> Dashboard
### Download Global Model
- Copy the desired global model id from the Dashboard (based on the score).
- Then go to the minio UI: https://localhost:9001
- the default login is username: fedn_admin, password: password (but you can change it in the fedn-repo under config/base-services.yaml, line 12-13)
- go to fedn-models -> Browse and download the model weights with the desired id.
To use the model for local inference, you could look at the notebook: SOON TO BE ADDED!

# Details
## Client configuration
If you wish to modify the training or the validation settings the relevant script is defined under the clients folder. Simple hypertuning of training and model can be made from the client/settings.yaml file:
```
# Parameters for model settings
binary_class: False
image_dimensions: 256, 256 # For downsampling (128,128)
Nfilter_start: 32
depth: 3

# Parameters for local training
batch_size: 8
epochs: 1

```
- Create a compute package. 
```
tar -czvf package.tar.gz client

```
For any modification regarding the model settings, initiate new seed model weights:
*Additional requirements: tensorflow 2, FEDn*

```
python create_seed_weights.py 
```
## Compute package
The compute package is a tarball of the client folder, it consists of all instructions to train, validate and communicate with the Combiner.
To read and write the model weights in the train.py and validate.py scripts FEDn is using a helper class (KerasHelper for keras models).
The fedn.yaml file is showing the entrypoints for the commands (train and validate in this exampple).
