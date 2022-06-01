import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from pathlib import Path


def writesentence(seedtext, howmanywords, longestsequencelength, model, tokenizer):

    for _ in range(howmanywords):
        tokenlist = tokenizer.texts_to_sequences([seedtext])[0]
        tokenlist = pad_sequences([tokenlist], maxlen=longestsequencelength - 1, padding='pre')
        predicted = np.argmax(model.predict(tokenlist), axis=1)
        outputword = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                outputword = word
                break
        seedtext += " " + outputword
    print(seedtext)


def formatinput(path):
    tokenizer = Tokenizer()

    data = open(Path(path)).read()

    linesofpoe = data.lower().split("\n")
    tokenizer.fit_on_texts(linesofpoe)
    totalwords = len(tokenizer.word_index) + 1

    print(tokenizer.word_index)
    print(totalwords)

    inputsequences = []
    for line in linesofpoe:
        tokenlist = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(tokenlist)):
            nGramSequence = tokenlist[:i + 1]
            inputsequences.append(nGramSequence)

    longestsequencelength = max([len(x) for x in inputsequences])
    inputsequences = np.array(pad_sequences(inputsequences, maxlen=longestsequencelength, padding='pre'))
    return inputsequences, totalwords, longestsequencelength, tokenizer


def retrainmodel(inputsequences, totalwords, longestsequencelength):
    xs = inputsequences[:, : -1]
    labels = inputsequences[:, -1]
    ys = tf.keras.utils.to_categorical(labels, num_classes=totalwords)

    model = Sequential()
    model.add(Embedding(totalwords, 240, input_length=longestsequencelength - 1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(totalwords, activation='softmax'))
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(xs, ys, epochs=60, verbose=1)
    # make sure to change the name of the model you are saving each time
    model.save('realtrainedmodel')



if __name__ == '__main__':
    path = "C:\\Users\\kenneth\\PycharmProjects\\poemwriter\\poems"
    inputsequences, totalwords, longestsequencelength, tokenizer = formatinput(path)
    newmodel = tf.keras.models.load_model('C:\\Users\\kenneth\\PycharmProjects\\poemwriter\\realtrainedmodel')
    writesentence("hello world", 10, longestsequencelength, newmodel, tokenizer)
