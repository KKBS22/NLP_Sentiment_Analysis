# import required packages
import os
import glob
import re
import csv
import shutil
import string
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import pickle
import nltk
import matplotlib
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


nltk.download('wordnet')
nltk.download('stopwords')

training_data = []
testing_data = []
stop_words = stopwords.words('english')
stemm_words = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


def load_data_to_csv(pathToData):
    DATA_PATH = {}
    WORKING_DIR = os.getcwd()
    os.chdir(WORKING_DIR)
    os.chdir(pathToData)
    # os.chdir("aclImdb/train")
    TRAIN_DIR = os.getcwd()
    LIST_OF_FILES = os.listdir(TRAIN_DIR)
    for a in LIST_OF_FILES:
        if a == 'pos':
            POS_PATH = os.path.join(TRAIN_DIR, a)
        if a == 'neg':
            NEG_PATH = os.path.join(TRAIN_DIR, a)
    DATA_PATH[1] = POS_PATH
    DATA_PATH[2] = NEG_PATH
    for key, values in DATA_PATH.items():
        if key == 1:
            CSV_NAME = 'IMDB_POS.csv'
            REVIEW_STAT = 'positive'
        else:
            CSV_NAME = 'IMDB_NEG.csv'
            REVIEW_STAT = 'negative'
        os.chdir(values)
        myFiles = glob.glob('*.txt')
        inc = 0
        with open(CSV_NAME, 'w', newline='', encoding='utf-8', errors='ignore') as f:
            header = ['review', 'index', 'rating', 'score']
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for a in myFiles:
                fileName = re.search('([\d]+)\_([\d]+)\.(txt)', a)
                file = open(a, "r", encoding='utf-8', errors='ignore')
                review = file.read()
                try:
                    filter_one_text = filter_text(review)
                except:
                    print('Unable to parse data for file :'+str(a))
                data = [filter_one_text, fileName.group(
                    1), fileName.group(2), REVIEW_STAT]
                # wiritng data into .csv file
                writer.writerow(data)
                inc = inc+1
                # print(inc)
    print("Finished Loading .txt files into .csv file")

    os.chdir("../")
    final_data = os.getcwd()
    pos_path_csv = os.path.join(DATA_PATH[1], 'IMDB_POS.csv')
    neg_path_csv = os.path.join(DATA_PATH[2], 'IMDB_NEG.csv')

    shutil.move(pos_path_csv, final_data)
    shutil.move(neg_path_csv, final_data)
    os.chdir(final_data)
    try:
        all_files = glob.glob(os.path.join(final_data, "IMDB_*.csv"))
        df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
        df_merged = pd.concat(df_from_each_file, ignore_index=True)
        df_merged.to_csv("IMDB_Movie.csv")
        # csv1 = pd.read_csv('IMDB_POS.csv')
        # csv2 = pd.read_csv('IMDB_NEG.csv')
        # df_merged = pd.concat([csv1, csv2], axis=0)
        # df_merged.to_csv("IMDB_data.csv")
        # print(merge_csv.shape)
    except:
        print("Unable to merge the CSV files")


def stopwords_removal(review):
    new_sentence = [word for word in review.split() if word not in stop_words]
    return " ".join(new_sentence)


def word_stemming(review):
    new_sentence = [stemm_words.stem(word) for word in review.split()]
    return " ".join(new_sentence)


def words_lemmatizer(review):
    new_sentence = [lemmatizer.lemmatize(word) for word in review.split()]
    return " ".join(new_sentence)


def filter_text(review):
    # Parsing reviews of each movie
    # parsing html content
    soup = BeautifulSoup(review, "html.parser")
    filter_one_text = soup.get_text()
    # convert caps to lower case
    filter_one_text = filter_one_text.lower()
    # remove brackets and other characters
    filter_one_text = re.sub(
        '\[[^]]*\]', '', filter_one_text)
    # remove quotes double quotes
    filter_one_text = re.sub(
        '[''"",,,]', '', filter_one_text)
    # removing whitespaces
    filter_one_text = re.sub('\w*\d\w*', '', filter_one_text)
    # removing next line
    filter_one_text = re.sub('\n', '', filter_one_text)
    # removing numbers
    filter_one_text = re.sub(
        '[^a-zA-z0-9\s]', '', filter_one_text)
    # removal of stop words
    filter_one_text = stopwords_removal(filter_one_text)
    # word stemming
    filter_one_text = word_stemming(filter_one_text)
    # word leammatization
    filter_one_text = words_lemmatizer(filter_one_text)
    return filter_one_text


def read_data():
    global training_data
    global testing_data
    CURRENT_DIR = os.getcwd()
    # os.chdir("../../aclImdb/train")
    # os.chdir("aclImdb/train")
    train_data = pd.read_csv('IMDB_Movie.csv', header=0).values
    #train_data = pd.read_csv('/content/IMDB_Movie.csv', header=0).values
    # print(len(train_data))
    split_val = round(0.8*len(train_data))
    indices = np.random.permutation(train_data.shape[0])

    training_idx, test_idx = indices[:split_val], indices[split_val:]
    training_data, testing_data = train_data[training_idx,
                                             :], train_data[test_idx, :]
    print(training_data.shape)
    print(testing_data.shape)
    pass


def sentiment_analyzer():
    max_length = 120
    word_vocab = 20000
    try:
        # Tokenization using tensor flow
        training_sentences = []
        training_labels = []
        testing_sentences = []
        testing_labels = []
        for a in training_data:
            training_sentences.append(a[1])
            if a[4] == 'negative':
                training_labels.append(0)
            if a[4] == 'positive':
                training_labels.append(1)
        for b in testing_data:
            testing_sentences.append(b[1])
            if b[4] == 'negative':
                testing_labels.append(0)
            if b[4] == 'positive':
                testing_labels.append(1)

        training_labels_final = np.array(training_labels)
        testing_labels_final = np.array(testing_labels)

        tokenizer = Tokenizer(num_words=word_vocab, oov_token="<OOV>")
        tokenizer.fit_on_texts(training_sentences)
        word_index = tokenizer.word_index
        # print(word_index)
        train_sequences = tokenizer.texts_to_sequences(training_sentences)
        train_padded = pad_sequences(
            train_sequences, maxlen=max_length, padding='post')

        test_sequences = tokenizer.texts_to_sequences(testing_sentences)
        test_padded = pad_sequences(
            test_sequences, maxlen=max_length, padding='post')

        # simple model
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Embedding(20000, 16, input_length=max_length),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(24, activation='relu'),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])
        # end

        # LSTM Model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(20000, 16, input_length=max_length),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # End

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.summary()
        num_epochs = 2
        history = model.fit(train_padded, training_labels_final, epochs=num_epochs, validation_data=(
            test_padded, testing_labels_final), verbose=2)

        # PLotting
        plt.clf()
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, (len(history_dict['loss']) + 1))
        plt.plot(epochs, loss_values, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss_values, 'r+-', label='Validation Loss')
        plt.title('Training v/s Validation Loss')
        plt.xlabel('No of Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("LossTrend", bbox_inches='tight', dpi=150)
        plt.show()

        plt.clf()
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        epochs = range(1, (len(history_dict['accuracy']) + 1))
        plt.plot(epochs, acc_values, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_acc_values, 'r+-', label='Validation Accuracy')
        plt.title('Training v/s Validation accuracy')
        plt.xlabel('No of Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("AccuracyTrend", bbox_inches='tight', dpi=150)
        plt.show()
        # saving the model
        model.save('NLP_Model.h5')
        with open('tokenizer_data.pkl', 'wb') as handle:
            pickle.dump(
                {'tokenizer': tokenizer, 'num_words': word_vocab, 'maxlen': max_length}, handle)

        # Testing model on unknown review
        sentence_test = ['The story of the daughter of an Indonesian politician who actually like did not contribute anything to this country Somehow it was appointed as a film maybe to promote her or her family political steps And in this film it seems like everyone is against Islam Films that sell religion like this are indeed the best at pretending to be victims Dont waste your time and money for this']
        sequences_test = tokenizer.texts_to_sequences(sentence_test)
        padded_test = pad_sequences(
            sequences_test, maxlen=max_length, padding='post')

        print(model.predict(padded_test))
    except:
        print('Unable to parse data')
    pass


def simple_filter():
    os.chdir(r'D:\UniversityOfWaterloo\Courses\657_ToolsOfIntelligentSystemDesign\assignment 3\Problem3\aclImdb_v1.tar\aclImdb_v1\aclImdb\train\pos')
    myFile = '12267_8.txt'
    file = open(myFile, "r", encoding='utf-8', errors='ignore')
    review = file.read()
    sentences = []
    try:
        print("###################### Before #####################")
        print(review)
        print("###################################################")
        filter_one_text = filter_text(review)
        print("###################### After #####################")
        print(filter_one_text)
        print("##################################################")
    except:
        print('Unable to parse data')
    pass


def main():
    # Function to be one time for loading the data into .csv file
    # os.chdir("../../aclImdb/train")
    os.chdir("aclImdb/train")
    LOAD_DATA_DIR = os.getcwd()
    load_data_to_csv(LOAD_DATA_DIR)
    read_data()
    sentiment_analyzer()
    # simple_filter()
    pass


if __name__ == '__main__':
    main()
