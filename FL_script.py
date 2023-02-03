import pandas as pd
import random
import spacy
import numpy as np
import math
import nltk
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score)
from tensorflow.keras import activations
from importlib import reload
import sys
import tensorflow_federated as tff
from importlib import reload
import warnings
import collections
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
import os
from collections import OrderedDict
from datetime import datetime
import tensorflow_privacy as tfp
import json


PREPROCESSED_DATASET_FILENAME="../abusive_100k_no_spams_preprocessed_all_v1.csv""../abusive_100k_no_spams_preprocessed_all_v1.csv"

MAX_FEATURES = 58439  # size of word index, set to the number of unique words in dataset
EMBEDDING_DIM = 200
MAX_LEN = 30  # average size of twitter text as used in the paper
# Local parameters
BATCH_SIZE = 10
EPOCHS = 7
RNN_CELL_SIZE = 128
# MAX_LEN = math.ceil(mean)  # calculate the mean of the texts size in the dataset
PREFETCH_BUFFER = 10
SHUFFLE_BUFFER = 100
# Federated Learning parameters
NUM_ROUNDS = 100
#CLIENT_LEARNING_RATE = 0.01
# SERVER_LEARNING_RATE = 0.5

#DP variables
noise_multiplier = 0.875
clients_per_round = 25

tff.backends.native.set_local_python_execution_context(clients_per_thread=10)


PREPROCESSED_DATASET_FILENAME="../abusive_100k_no_spams_preprocessed_all_v1.csv"

def split_dataset(file):
    # read dataset file and create lists
    df = pd.read_csv(file)
    
    df.dropna()
    
    tweet_texts, labels = df["text"],df["label"]
    
    print("ones - inappropriate :", len(df[df['label']==1]))
    print("zeros - normal:",  len(df[df['label']==0]))
    print("dataset size= ",len(df))

    dataset_size = len(df)

    no = list()
    for i in range(0, len(tweet_texts)):
        no.append(i)

    df = pd.DataFrame(list(zip(no, tweet_texts, labels)), columns=['no', 'text', 'label'])

    normal_df = df[df['label'] == 0]
    normal_index = set(normal_df.no.unique())

    aggressive_df = df[df['label'] == 1]
    aggressive_index = set(aggressive_df.no.unique())

    #set the test set size
    test_set_size = math.ceil(0.1 * dataset_size)

    #set the test size ratio for inappropriate 8% and norm
    NUM_OF_AG_DATA = math.ceil(test_set_size * 0.08)
    NUM_OF_NORM_DATA = test_set_size - NUM_OF_AG_DATA

    print("Test set:\n num of aggressive: ", NUM_OF_AG_DATA, "\n num of normal: ", NUM_OF_NORM_DATA, "\n total:",
          (NUM_OF_AG_DATA + NUM_OF_NORM_DATA))

    #choose randomly from the inappropriate and normal pool to create the test set
    test_index_norm = set(random.sample(normal_index, NUM_OF_NORM_DATA))
    test_index_agr = set(random.sample(aggressive_index, NUM_OF_AG_DATA))

    test_index = test_index_agr.union(test_index_norm)

    #delete the selected from the pool
    normal_index = normal_index - test_index_norm
    aggressive_index = aggressive_index - test_index_agr

    random.shuffle(list(test_index))

    #create a dataframe for the test set
    test_df = pd.DataFrame()
    for i in list(test_index):
        row = df[df['no'] == i]
        frames = [test_df, row]
        test_df = pd.concat(frames)

    test_set_file = open('rep'+str(sys.argv[1])+"/test_set.csv", "a+")
    test_df.to_csv(test_set_file,index=False)  

    #test_set_file.write("text,label\n")
    #for index, row in test_df.iterrows():
    #    test_set_file.write(str(row['text']) + "," + str(row['label']) + "\n")
    #test_set_file.close()

    train_index = normal_index.union(aggressive_index)
    random.shuffle(list(train_index))
    train_df = pd.DataFrame()
    for i in list(train_index):
        row = df[df['no'] == i]
        frames = [train_df, row]
        train_df = pd.concat(frames)

    train_set_file = open('rep'+str(sys.argv[1])+"/train_set.csv", "a+")
    train_df.to_csv(train_set_file,index=False)  

    #train_set_file.write("text,label\n")
    #for index, row in train_df.iterrows():
    #    train_set_file.write(str(row['text']) + "," + str(row['label']) + "\n")
    #train_set_file.close()

split_dataset(PREPROCESSED_DATASET_FILENAME)
exit()

def federated_clients():
    df = pd.read_csv('rep'+str(sys.argv[1])+'/train_set.csv',)


    no = list()
    for i in range(0, len(df)):
        no.append(i)
    df['no'] = no


    normal_df = df[df['label'] == 0]
    normal_index = set(normal_df.no.unique())

    aggressive_df = df[df['label'] == 1]
    aggressive_index = set(aggressive_df.no.unique())



    print("Train set:\n num of aggressive: ", len(aggressive_index), "\n num of normal: ", len(normal_index),
          "\n total:",
          (len(normal_index) + len(aggressive_index)))

    num_agr = 50 
    num_norm = 50 

    NUM_OF_CLIENTS= min(math.floor(len(normal_index)/ num_norm),math.floor(len(aggressive_index)/ num_agr))
    print("THE NUMBER OF SIMULATED CLIENTS: ",NUM_OF_CLIENTS)

    clients_df = pd.DataFrame(columns=['client', 'text', 'label'])
    for i in range(0, NUM_OF_CLIENTS):
        #print("client ", i)
        client_data = pd.DataFrame()
       # if (i == NUM_OF_CLIENTS - 1):
        select_normal = set(random.sample(normal_index, num_norm))#len(normal_index)))
        select_agressive = set(random.sample(aggressive_index, num_agr))#len(aggressive_index)))
        user_index = select_normal.union(select_agressive)
        random.shuffle(list(user_index))
        normal_index = normal_index - select_normal
        aggressive_index = aggressive_index - select_agressive
        #print(len(select_normal), len(select_agressive))
        for j in user_index:
            row = df[df['no'] == j]
            frames = [client_data, row]
            client_data = pd.concat(frames)
        for index, row in client_data.iterrows():
            d = {'client': ("client_" + str(i)), 'text': str(row['text']), 'label': float(row['label'])}
            clients_df = clients_df.append(d, ignore_index=True)

    return clients_df

# uncomment to create the clients binary dataset csv
#test_df,valid_df,clients_df=federated_clients()



def build_complex_model():#print_summary=False, text_frozen=False, other_frozen=False):
    return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(MAX_LEN,)),
            tf.keras.layers.Embedding(input_dim=text_embedding.shape[0],
                                      output_dim=text_embedding.shape[1],
                                      weights=[text_embedding],
                                      input_length=MAX_LEN,
                                      mask_zero=True, trainable=False),
            tf.keras.layers.GRU(units=RNN_CELL_SIZE, implementation=2, recurrent_dropout=0.5,
                                return_sequences=False),
            tf.keras.layers.Dense(1, activation='sigmoid')
             ])



warnings.filterwarnings('ignore')
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)




def loadGloveModel():
    print("Loading Glove Model")
    files = ["../glove.twitter.27B." + str(EMBEDDING_DIM) + "d.txt"]
    for file in files:
        f = open(file, 'r', encoding='utf-8')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        f.close()
    print(len(gloveModel), " words loaded!")
    return gloveModel


def create_lists_evaluate(set,df):
    # fields = [ 'text', 'label']
    # df = pd.read_csv('abusive_client_binary_dataset_'+str(client_number)+'_clients_'+str(set)+'_set.csv', sep=',', usecols=fields)
    # df.dropna()
    print(set+" dataset length:", len(df))

    labels = list()
    tweet_texts = list()

    for index, row in df.iterrows():
        labels.append(row["label"])
        tweet_texts.append(row['text'])

    return tweet_texts, labels


def create_lists(df):
    #fields = ['client', 'text', 'label']
    #df = pd.read_csv('abusive_client_binary_dataset_'+str(client_number)+'_clients_train_set.csv', sep=',', usecols=fields)
    #df.dropna()
    print("Dataset length:", len(df))

    labels = list()
    tweet_texts = list()
    user_ids = list()

    for index, row in df.iterrows():
        labels.append(row["label"])
        tweet_texts.append(row['text'])
        user_ids.append(row['client'])

    return tweet_texts, labels, user_ids

#split original dataset to train and set
split_dataset()

clients_df=federated_clients()
tweet_texts, labels, user_ids = create_lists(clients_df)

print("tweets len:", len(tweet_texts), "labels len:", len(labels))
print("Abusive/Hateful number(class 1): ", labels.count(float(1)))
print("Normal number(class 0): ", labels.count(float(0)))

tokenizer = Tokenizer(num_words=MAX_FEATURES)
# create a word index vocabulary for each word in the dataset

tokenizer.fit_on_texts(tweet_texts)
word_index = tokenizer.word_index

# load pretrained Embedding
gloveModel = loadGloveModel()

text_embedding = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
found = 0
not_found = 0
# embedding_matrix = np.zeros((len(word_index) + 1, 25))
for word, i in word_index.items():
    if i >= (len(word_index) + 1):
        continue
    embedding_vector = gloveModel.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        text_embedding[i] = embedding_vector
        found += 1
    else:
        not_found += 1


#print("Found: ", found, " Not found: ", not_found)

list_tokenized_train = tokenizer.texts_to_sequences(tweet_texts)
x = pad_sequences(list_tokenized_train, maxlen=MAX_LEN, padding="post")
y = labels

test_df = pd.read_csv('rep'+str(sys.argv[1])+'/test_set.csv')
x_test,y_test=create_lists_evaluate("test", test_df)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post")

# x_valid,y_valid=create_lists_evaluate("valid", valid_df)
# x_valid=tokenizer.texts_to_sequences(x_valid)
# x_valid=pad_sequences(x_valid, maxlen=MAX_LEN, padding="post")


METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    #keras.metrics.Precision(name='precision'),
    #keras.metrics.Recall(name='recall'),
    #keras.metrics.AUC(name='auc'),
]


# create federated learning dataset
client_train_dataset = collections.OrderedDict()
df = pd.DataFrame(list(zip(user_ids, x, y)),
                  columns=['uid', 'text', 'label'])
# print(df)
user_ids = df.uid.unique()
# print("Number of user ids in trainning set:", len(user_ids))
NUM_OF_SIMULATED_CLIENTS = len(user_ids)
for i in user_ids:
    # print("user_id= ",i)
    client_name = "client_" + str(i)

    # select all rows where client id == i
    client_data = df[df['uid'] == i]

    text_list = client_data['text'].tolist()
    label_list = client_data['label'].tolist()
    #print(client_name,label_list.count(0),label_list.count(1))
    # print(len(label_list), len(text_list))

    text_list = np.array(text_list)

    label_list = np.array(label_list)

    # print(len(label_list), len(text_list))

    data = collections.OrderedDict((('x', text_list), ('y', label_list)))
    client_train_dataset[client_name] = data


#train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

sample_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
#sample_element = next(iter(sample_dataset))

def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=element['x'],
            y=tf.reshape(element['y'],[-1,1])
        )

    return dataset.repeat(EPOCHS)\
        .shuffle(SHUFFLE_BUFFER)\
        .batch(BATCH_SIZE)\
        .map(batch_format_fn)\
        .prefetch(PREFETCH_BUFFER)


preprocessed_example_dataset = preprocess(sample_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))



def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]


# choose randomly N client ids and add them in a list for a federated round
def random_selected_federated_data(train_dataset, N):
    client_ids = random.sample(train_dataset.client_ids, N)
    # print(client_ids)
    # make federate data with the selected client ids
    federated_train_data = make_federated_data(train_dataset, client_ids)
    return federated_train_data


def model_fn():
    keras_model = build_complex_model()
    return tff.learning.from_keras_model(keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

    



aggregation_factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
      noise_multiplier, clients_per_round)

iterative_process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),#learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(), model_aggregator=aggregation_factory)#learning_rate=1.0))
#older TFF version
# iterative_process = tff.learning.build_federated_averaging_process(
#     model_fn)
state = iterative_process.initialize()

history = list()
history_val = list()
valid_test = list()

#no_clients_in_fl = 21

#uncomment to have the same clients in all fl rounds
#train_data = random_selected_federated_data(train_dataset,no_clients_in_fl)

#print("Train clients: ", NUM_OF_SIMULATED_CLIENTS)#, "Test clietns: ", NUM_OF_TEST_CLIENTS)



# We use Poisson subsampling which gives slightly tighter privacy guarantees
# compared to having a fixed number of clients per round. The actual number of
 # clients per round is stochastic with mean clients_per_round.

total_clients = 628
sampling_prob = clients_per_round / total_clients

out_file = open('rep'+str(sys.argv[1])+"/results.txt", "a+", encoding='utf-8')
out_file.write("Type\tAccuracy\tLoss\n")
out_file.close()

out_file = open('rep'+str(sys.argv[1])+"/round_test_results.txt", "a+", encoding='utf-8')
out_file.write("Repetition\tRound_Number\tAccuracy\tPrecision\tRecall\tAUC\tF1-score\tLoss\n")
out_file.close()

for round_num in range(0, NUM_ROUNDS):

    # Sample clients for a round. Note that if your dataset is large and
    # sampling_prob is small, it would be faster to use gap sampling.
    x = np.random.uniform(size=total_clients)
    sampled_clients = [
        train_dataset.client_ids[i] for i in range(total_clients)
        if x[i] < sampling_prob]
    print(sampled_clients)
    train_data = make_federated_data(train_dataset, sampled_clients)
        # [
        # train_dataset.create_tf_dataset_for_client(client)
        # for client in sampled_clients]



    #uncomment to have randomly selected(different) clients in each fl round
    #train_data=random_selected_federated_data(train_dataset,no_clients_in_fl)
    start_time = datetime.now()
    start_time_pr = start_time.strftime("%H:%M:%S")
    print("Round "+str(round_num)+ " started at = ", start_time_pr)

    #state, metrics = iterative_process.next(state, train_data)

    result = iterative_process.next(state, train_data)
    state = result.state
    metrics = result.metrics

    end_time = datetime.now()
    end_time_pr = end_time.strftime("%H:%M:%S")
    print("Round "+str(round_num)+" completed at = ", end_time_pr)

    total_round_execution_time=end_time-start_time

    out_file_time = open('rep'+str(sys.argv[1])+"/round_times.txt", "a+", encoding='utf-8')
    out_file_time.write(str(round_num)+"\t"+str(total_round_execution_time)+"\n")
    out_file_time.close()



    evaluation = tff.learning.build_federated_evaluation(model_fn)
    model_weights = iterative_process.get_model_weights(state)

    train_metrics = evaluation(model_weights, train_data)
    print("Train metrics: \n round {:2d}, metrics={}".format(round_num, train_metrics))
    # print("Test metrics: \n round {:2d}, metrics={}".format(round_num, test_metrics))
    accuracy=train_metrics["eval"]["accuracy"]
    loss=train_metrics["eval"]["loss"]


    train_metrics = OrderedDict([('accuracy', float(accuracy)),
                                 ('loss', float(loss)),])
                                 
    history.append(train_metrics)


    out_file = open('rep' + str(sys.argv[1]) + "/results.txt", "a+", encoding='utf-8')
    out_file.write(
        "training\t" + str(accuracy) + "\t"+ str(loss) + "\n")
    out_file.close()


    # to validate the model on centralized data
    keras_model = build_complex_model()#other_frozen=True)
    model_weights = iterative_process.get_model_weights(state)
    model_weights.assign_weights_to(keras_model)


    opt = keras.optimizers.Adam()
    keras_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=METRICS)

    y_pred_class = keras_model.predict(x_test)
    y_pred_class=y_pred_class.tolist()

    accuracy = tf.keras.metrics.BinaryAccuracy()
    accuracy.update_state(y_test,y_pred_class)
    accuracy=accuracy.result().numpy()
    print('Valid accuracy: ', accuracy)

    y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss=loss(y_test, y_pred_class).numpy()
    print('Valid loss: ', loss)


    val_metrics = OrderedDict([('accuracy', float(accuracy)),
                               ('loss', float(loss))])

   

    history_val.append(val_metrics)

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        log_loss,
        roc_auc_score,
        accuracy_score
    )


    if round_num % 10 == 0:
        # to test the model on centralized data
        keras_model = build_complex_model()  # other_frozen=True)
        model_weights = iterative_process.get_model_weights(state)
        model_weights.assign_weights_to(keras_model)

        opt = keras.optimizers.Adam()  # learning_rate=CLIENT_LEARNING_RATE)
        keras_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=METRICS)

       


        # make the predictions for testing evaluation
        y_pred_class = keras_model.predict(x_test)
        cf_matrix = confusion_matrix(y_test, y_pred_class.round())

        # compute the evaluation metrics
        recall = recall_score(y_test, y_pred_class.round(), average="weighted")
        precision = precision_score(y_test, y_pred_class.round(), average="weighted")
        f1_score = f1_score(y_test, y_pred_class.round(), average="weighted")
        auc = roc_auc_score(y_test, y_pred_class.round(), average="weighted")
        accuracy = accuracy_score(y_test, y_pred_class.round())
        loss = log_loss(y_test, y_pred_class)

        

        print("Testing Performance:\n")
        print("Confusion Matrix: ", cf_matrix)
        print("Accuracy:\t", str(accuracy))
        print("Precision:\t", precision)
        print("Recall:\t", recall)
        print("F1_score:\t", f1_score)
        print("AUC:\t", str(auc))
        print('loss:\t', str(loss))

        out_file = open('rep'+str(sys.argv[1])+"/round_test_results.txt", "a+", encoding='utf-8')
        out_file.write(str(sys.argv[1]) +"\t"+str(round_num)+ "\t" + str(accuracy) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(
        auc) + "\t" + str(f1_score) + "\t" + str(loss) + "\n")
    out_file.close()

#write json object to a file
with open('rep'+str(sys.argv[1])+'/training_results.txt', 'w') as fout:
    json.dump(history, fout)
fout.close()
#convert list of metrics to json string
history_val=json.dumps(history_val)
#convert json string to json_object/dict
history_val=json.loads(history_val)
with open('rep'+str(sys.argv[1])+'/valid_results.txt', 'w') as fout:
    json.dump(history_val, fout)
fout.close()

# to test the model on centralized data
keras_model = build_complex_model()#other_frozen=True)
model_weights = iterative_process.get_model_weights(state)
model_weights.assign_weights_to(keras_model)


opt = keras.optimizers.Adam()#learning_rate=CLIENT_LEARNING_RATE)
keras_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=METRICS)


#make the predictions for evaluation
y_pred_class = keras_model.predict(x_test)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    accuracy_score
)
cf_matrix = confusion_matrix(y_test, y_pred_class.round())

#compute the evaluation metrics

recall = recall_score(y_test, y_pred_class.round(),average="weighted")
precision = precision_score(y_test, y_pred_class.round(),average="weighted")
f1_score = f1_score(y_test,y_pred_class.round(),average="weighted")
auc = roc_auc_score(y_test, y_pred_class.round(), average="weighted")
accuracy = accuracy_score(y_test, y_pred_class.round())
loss = log_loss(y_test, y_pred_class)



print("Testing Performance:\n")
print("Confusion Matrix: ", cf_matrix)
print("Accuracy:\t",str(accuracy))
print("Precision:\t", precision)
print("Recall:\t", recall)
print("F1_score:\t",f1_score)
print("AUC:\t",str(auc) )
print('loss:\t',str(loss))

out_file = open("final_test_results.txt", "a+", encoding='utf-8')
out_file.write("Repetition\tAccuracy\tPrecision\tRecall\tAUC\tF1-score\tLoss\n")
out_file.write(str(sys.argv[1])+"\t"+str(accuracy)+"\t"+str(precision)+"\t"+str(recall)+"\t"+str(auc)+"\t"+str(f1_score)+"\t"+str(loss)+"\n")
out_file.close()


#store true labels and predicted

y_pred_class=y_pred_class.tolist()
y_test=y_test.tolist()

data = OrderedDict([('y_true',y_test),
                   ('y_pred',y_pred_class)])

data=json.dumps(data)
with open('rep'+str(sys.argv[1])+'/test_model_labels.txt', 'w') as outfile:
    json.dump(data, outfile)
outfile.close()

# Cross Validation Classification Accuracy

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
mpl.rcParams["figure.figsize"] = (12, 12)


def plot_metrics(train_metrics, valid_metrics):
    metrics = [
        "loss","accuracy"]
    for n, metric in enumerate(metrics):

        met = list()
        for i in train_metrics:
            met.append(float(i[metric]))
        name = metric.replace("_", " ").capitalize()
        plt.subplot(3, 2, n + 1)
        met_valid = list()
        for valid in valid_metrics:
            met_valid.append(float(valid[metric]))
        name = metric.replace("_", " ").capitalize()
        plt.subplot(3, 2, n + 1)
        x = np.arange(0, NUM_ROUNDS, 1)

        plt.plot(
            x,
            met,
            color=colors[0],
            label="Train",
        )
        plt.plot(
            x,
            met_valid,
            color=colors[1],
            label="Valid",
        )
       
        plt.xlabel("Federated Learning Round")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1] * 1.2])
        elif metric == "accuracy":
            plt.ylim([0, 1])
        elif metric == "precision":
            plt.ylim([0, 1])
        elif metric == "recall":
            plt.ylim([0, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
        plt.savefig("training_plots_"+str(sys.argv[1])+".png")


plot_metrics(history, history_val)


plt.clf()
plt.close()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
mpl.rcParams["figure.figsize"] = (8, 8)
group_names = [' TN','FP','FN','TP']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
#group_percentages = ['{0:.2%}'.format(value) for value in
#                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n' for v1, v2 in
          zip(group_names,group_counts)] #,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
svm=sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

#svm=sns.heatmap(cf_matrix, annot=True)
figure = svm.get_figure()
figure.savefig("conf_matrix_"+str(sys.argv[1])+".png",dpi=400)










