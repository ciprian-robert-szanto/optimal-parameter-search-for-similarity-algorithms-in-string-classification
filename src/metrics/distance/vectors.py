import nltk
import numpy as np
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from utils.preprocess import _preprocess

nltk.download('punkt')


def _tfidf_cosine_similarity(input_string, dataset_strings):
    """Distance metric using TfidfVectorizer."""
    # Prepare vectorizer
    # analyze='char' -> the dataset contains small sentences
    # ngram_range=(1, 2) -> for catching typos
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 2)
    )

    # Vectorize all the strings
    all_data = dataset_strings + [input_string]
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Get the input vector and all the dataset vectors
    input_vector = tfidf_matrix[-1]
    dataset_vectors = tfidf_matrix[:-1]

    # Compute similarity
    cosine_ndarray = cosine_similarity(input_vector, dataset_vectors)
    return cosine_ndarray.flatten()


def _tfidf_euclidean_distance(input_string, dataset_strings):
    """Distance metric using TfidfVectorizer."""
    # Prepare vectorizer
    # analyze='char' -> the dataset contains small sentences
    # ngram_range=(1, 2) -> for catching typos
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 2)
    )

    # Vectorize all the strings
    all_data = dataset_strings + [input_string]
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Get the input vector and all the dataset vectors
    input_vector = tfidf_matrix[-1]
    dataset_vectors = tfidf_matrix[:-1]

    # Compute distance
    euclidean_ndarray = euclidean_distances(input_vector, dataset_vectors)
    flattened = euclidean_ndarray.flatten()
    return 1 - flattened / flattened.max()


def _fasttext_cosine_similarity(text1, text2, fasttext_model):
    """Distance metric using FastText and cosine_similarity."""
    tokens1 = set(_preprocess(text1))
    tokens2 = set(_preprocess(text2))
    model_keys = set(fasttext_model.wv.key_to_index.keys())

    intersect1 = tokens1 & model_keys
    intersect2 = tokens2 & model_keys

    vector1 = np.mean([fasttext_model.wv[token]
                      for token in intersect1], axis=0)
    vector2 = np.mean([fasttext_model.wv[token]
                      for token in intersect2], axis=0)

    return cosine_similarity([vector1], [vector2])[0][0]


def _fasttext(input_string, dataset_strings):
    all_data = [_preprocess(doc) for doc in dataset_strings + [input_string]]
    fasttext_model = FastText(
        sentences=all_data, vector_size=100, window=3, min_count=1, epochs=10)

    return np.array([_fasttext_cosine_similarity(
        input_string, text, fasttext_model) for text in dataset_strings])


def _vector_similarity_controller(measure, input_string, dataset_strings):
    # Select the function to use
    match measure:
        case 'tfidf_cosine':
            trigger_function = _tfidf_cosine_similarity

        case 'tfidf_euclidean':
            trigger_function = _tfidf_euclidean_distance

        case 'fasttext_cosine':
            trigger_function = _fasttext

        case _:
            raise ValueError(f"Unknown similarity measure: {measure}")

    return trigger_function(input_string, dataset_strings)
