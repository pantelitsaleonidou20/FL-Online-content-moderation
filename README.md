# FL Online content moderation

This is the repository of the scripts used for the experimental evaluation of the work "Privacy–Preserving Online Content Moderation: A Federated
Learning Use Case".

Steps:

# 1. Access the Datasets
The datasets we used in our evaluation can be access after sending a request for access to the owners of the datasets e.g. for
the Abusive Dataset send an access request using the following link https://zenodo.org/record/3706866#.Y90_X-xBxJU.

# 2. Data preprocessing
The script preprocess_dataset.py contains the code used for preprocessing the tweet texts to prepare the data for the training task.

# 3. Differential Private Federated Learing Simulation
The script FL_scrip.py contains the code used for:

1) splitting the preprocessed dataset to train and test sets

2) distribute the train data to a number of clients -- simulate the federated train set

3) prepare the settings for the DP-FL simulation

4) execute the DP-FL training

5) evalaute the global model

For (3) and (4) we follow the following tensorflow federated tutorials: 
1. https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification (for Federated Learning simulation)
2. https://www.tensorflow.org/federated/tutorials/federated_learning_with_differential_privacy (for Differential Privacy)




