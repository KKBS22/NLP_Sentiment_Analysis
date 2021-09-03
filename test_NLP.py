# import required packages
import os
import tensorflow as tf
import pickle
import pandas as pd

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train_NLP import filter_text
from train_NLP import load_data_to_csv

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


def main():
    WORKING_DIR = os.getcwd()
    # os.chdir("../../aclImdb/test")
    os.chdir("aclImdb/test")
    DATA_DIR = os.getcwd()
    load_data_to_csv(DATA_DIR)
    test_data = pd.read_csv('IMDB_Movie.csv', header=0).values
    os.chdir(WORKING_DIR)
    os.chdir("models")
    NLP_Model_Test = tf.keras.models.load_model('36_NLP_model.h5')
    with open("tokenizer_data.pkl", 'rb') as f:
        data = pickle.load(f)
        tokenizer = data['tokenizer']
        num_words = data['num_words']
        maxlen = 120
    sentence_test = []
    status = 0
    accuracy = 0
    for a in test_data:
        sentence_test.append(a[1])
        #sentence_filtered = filter_text(sentence_test[0])
        #sentence_test[0] = sentence_filtered
        sequences_test = tokenizer.texts_to_sequences(sentence_test)
        test_padded = pad_sequences(sequences_test, maxlen=120, padding='post')
        if a[4] == 'positive':
            status = 1
        else:
            status = 0
        testVal = NLP_Model_Test.predict(test_padded)
        preVal = testVal[0][0]
        predicted_score = round(preVal)
        if predicted_score == status:
            accuracy = accuracy+1
        sentence_test.clear()
    correctness = (accuracy/25000)*100
    print("Accuracy of the model for sentiment analysis is: "+str(correctness))


if __name__ == "__main__":
    main()

    # 2. Load your testing data

    # 3. Run prediction on the test data and print the test accuracy
