from math import floor
from sys import argv

import enum
import re

import nltk
import numpy as np
import typing
import unicodecsv as csv
import tensorflow as tf
from tensorflow import keras

from stopwords import STOPWORDS


class ColumnIndex(enum.Enum):
    """These are the columns in the CSV."""
    deal_id = 0
    title = enum.auto()
    standard_price = enum.auto()
    discount_price = enum.auto()
    has_regions = enum.auto()
    category_id = enum.auto()
    site_name = enum.auto()
    category_name = enum.auto()


def tokenize_title(title: str) -> typing.Set[str]:
    """Convert a title to a set of tokens."""
    porter = nltk.PorterStemmer()
    letter_only_re = re.compile('^[A-z]{2,}$')
    cleaned_title = title.lower().replace("'", '')
    cleaned_tokens = set([porter.stem(token) for token in nltk.wordpunct_tokenize(cleaned_title)
                          if (token not in STOPWORDS) and letter_only_re.match(token)])
    return cleaned_tokens


def main(product_csv_path: str):
    words = []
    word_lookup = {}
    word_count = {}

    tokenized_titles = []
    categories_names = []

    deal_category_names = []

    title_original = []

    deal_category_indexes = []

    with open(product_csv_path, 'rb') as f:
        reader = csv.reader(f)

        next(reader, None)  # Skip Header Line

        for line in reader:
            title = line[ColumnIndex.title.value]

            title_tokens = tokenize_title(title)

            title_original.append(title)

            for token in title_tokens:
                if token not in words:
                    words.append(token)
                    word_count[token] = 0

                word_count[token] += 1

            tokenized_titles.append(title_tokens)

            category_name = line[ColumnIndex.category_name.value]

            if category_name not in categories_names:
                categories_names.append(category_name)

            deal_category_names.append(category_name)

    # Input layer size could be reduced if only feeding in words that occur in more than 1 product
    # This would reduce memory usage too BUT decreases accuracy by ~1%
    common_words = words  # [word for word in words if word_count[word] > 1]

    for word in common_words:
        # convert to a dictionary of word index lookups, faster than scanning for the index in an array
        word_lookup[word] = len(word_lookup)

    title_vectors = build_title_vectors(tokenized_titles, word_lookup, words)

    for category_name in deal_category_names:
        deal_category_indexes.append(categories_names.index(category_name))

    assert len(deal_category_indexes) == len(title_vectors)
    assert len(title_vectors) == len(title_original)

    # Convert to numpy arrays that Keras requires
    title_vectors = np.array(title_vectors)
    deal_category_indexes = np.array(deal_category_indexes)

    # use 80% for training
    training_size = floor(0.8 * len(deal_category_indexes))

    training_title_vectors = title_vectors[:training_size]
    training_categories = deal_category_indexes[:training_size]

    # use other 20% for testing the model
    test_title_vectors = title_vectors[training_size:]

    # split the original titles and categories in the same way for reference
    original_test_titles = title_original[training_size:]
    test_categories = deal_category_indexes[training_size:]

    # build the model
    model = build_model(categories_names, training_categories, training_title_vectors, words)

    # test the model against the test data
    test_loss, test_accuracy = model.evaluate(test_title_vectors, test_categories)

    print('Test accuracy:', test_accuracy)

    # Generate an array of prediction vectors
    predictions = model.predict(test_title_vectors)

    for i, prediction in enumerate(predictions):
        # each prediction is an array with probabilities for each category at that index. Take the index of the maximum
        max_prediction = int(np.argmax(prediction))

        if max_prediction != test_categories[i]:  # print out only the ones which category don't match
            print('{}\n- {} - {}\n'.format(original_test_titles[i], categories_names[max_prediction],
                                           categories_names[test_categories[i]]))


def build_title_vectors(tokenized_titles: typing.List[typing.Set[str]], word_lookup: typing.Dict[str, int],
                        words: typing.List[str]) -> typing.List[typing.List[int]]:
    """Convert each tokenized title (set of str) to an integer vector."""
    title_vectors = []
    for title_tokens in tokenized_titles:
        title_vector = convert_title_tokens_to_vector(title_tokens, word_lookup, words)
        title_vectors.append(title_vector)
    return title_vectors


def convert_title_tokens_to_vector(title_tokens: typing.Set[str], word_lookup: typing.Dict[str, int],
                                   words: typing.List[str]) -> typing.List[int]:
    """Take a title that has been converted to tokens (list of str) to an integer vector."""
    title_vector = [0] * len(words)
    for token in title_tokens:
        if token in word_lookup:
            title_vector[word_lookup[token]] = 1
    return title_vector


def build_model(categories_names: typing.List[str], training_categories: np.ndarray, training_titles: np.ndarray,
                words: typing.List[str]) -> keras.Sequential:
    """Build the keras neural network and train it."""
    model = keras.Sequential([
        keras.layers.Dense(len(words), tf.nn.relu),
        keras.layers.Dense(len(categories_names), tf.nn.softmax)
    ])
    model.compile(optimizer='nadam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    try:
        model.fit(training_titles, training_categories, epochs=5, batch_size=32)
    except Exception:
        raise  # just here so I can put a breakpoint
    return model


if __name__ == '__main__':
    main(argv[1])
