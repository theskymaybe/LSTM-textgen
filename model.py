import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Activation,
    Dense, Dropout, LSTM)
from keras.models import Sequential
from keras.optimizers import RMSprop


def sample(predictions, diversity):
    predictions = numpy.asarray(predictions).astype('float64')
    predictions = numpy.exp(numpy.log(predictions) / diversity)
    predictions = predictions / numpy.sum(predictions)
    probabilities = numpy.random.multinomial(1, predictions, 1)
    return numpy.argmax(probabilities)


def setup_model(vectorization):
    lstm_model = Sequential()
    layer = LSTM(input_shape=(vectorization.words_per_sample(),
                              vectorization.unique_word_count()),
                 units=256)
    lstm_model.add(layer)
    lstm_model.add(Dropout(rate=0.2))
    lstm_model.add(Dense(units=vectorization.unique_word_count()))
    lstm_model.add(Activation('softmax'))
    return lstm_model


def compile_model(lstm_model):
    lstm_model.compile(loss='categorical_crossentropy',
                       optimizer=RMSprop(lr=0.01))


def train_model(lstm_model, vectorization, weights_filename_template,
                max_epoch):
    results_callback = ModelCheckpoint(
        filepath=weights_filename_template,
        monitor='loss',
        verbose=1,
        save_best_only=False,
        mode='min')
    lstm_model.fit(
        x=vectorization.training_data(),
        y=vectorization.target_data(),
        batch_size=128,
        epochs=max_epoch,
        verbose=1,
        callbacks=[results_callback],
        initial_epoch=0)


def generate_text(words, lstm_model, vectorization, seed, diversity):
    words_per_sample = vectorization.words_per_sample()
    generated_sequence = []
    for i in range(vectorization.words_per_sample()):
        generated_sequence.append(vectorization.encode_word(words[seed + i]))
    for i in range(400):
        training_data = numpy.zeros(shape=(
            1, vectorization.words_per_sample(), vectorization.unique_word_count()))
        # Take words_per_sample last words of the generated text and
        # generate a new word, then append it to the generated text
        for j, training_data_word in enumerate(generated_sequence[-words_per_sample:]):
            training_data[0, j, training_data_word] = 1
        predictions = lstm_model.predict(training_data)[0]
        generated_sequence.append(sample(predictions, diversity))
    return vectorization.decode_words(generated_sequence)
