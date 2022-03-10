# FEDn-ImageSegmentation-tutorial



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
Constrains: \[NR_OF_CLIENTS] **x** **(** \[TRAINING_SUBJECTS_PER_CLIENTS] **+** \[VALIDATION_SUBJECTS_PER_CLIENTS] **)** **$\leq$** 369

- $x + y$
- $x - y$
- $x \times y$ 
- $x \div y$
- $\dfrac{x}{y}$
- $\sqrt{x}$

## Setting up pseudo deployment

- clone the FEDn repo:
```
git clone https://github.com/scaleoutsystems/fedn.git
```
(use develop branch)
- Follow the instructions on the README.md for how to set up the base, reducer and combiner container.
(For the moment use these commands to set up the Reducer and Combiner:)
```
docker-compose -f config/reducer-dev.yaml -f config/private-network.yaml up 
```
```
docker-compose -f config/combiner-dev.yaml -f config/private-network.yaml up 
```


- Open the Reducer UI: https://localhost:8090
- Choose keras-helper and Package: package.tar.gz (from this repository).
- Set the initial model: initial_weights.npz (from this repository)
- Go to Network and download client config to this folder.
- Set up the client:
```
docker-compose -f docker-compose.yaml -f private-network.yaml up 
```
- Start the training from the Reducer UI -> Control:

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
