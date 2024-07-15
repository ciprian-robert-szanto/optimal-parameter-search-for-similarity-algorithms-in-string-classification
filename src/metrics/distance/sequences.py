import numpy as np
from nltk.metrics.distance import jaro_similarity, jaro_winkler_similarity

from utils.preprocess import _preprocess


def _sequence_similarity_controller(measure, input_string, dataset_strings):
    # Preprocess the input
    input_tokens = list(_preprocess(input_string))

    # Select the function to use
    match measure:
        case 'jaro':
            trigger_function = _jaro_similarity

        case 'jaro_winkler':
            trigger_function = _jaro_winkler_similarity

        case _:
            raise ValueError(f"Unknown similarity measure: {measure}")

    # Create an array to add each similarity
    similarities = []
    for text in dataset_strings:
        # Preprocess each phrase
        tokens = list(_preprocess(text))
        # Compute similarity
        similarity_coef = trigger_function(input_tokens, tokens)
        similarities.append(similarity_coef)

    return np.array(similarities)


def _jaro_similarity(set1, set2):
    """The Jaro distance between is the min no. of single-character
    transpositions required to change one word into another."""
    return jaro_similarity(set1, set2)


def _jaro_winkler_similarity(set1, set2, p=0.2, max_l=4):
    """The Jaro distance between is the min no. of single-character
    transpositions required to change one word into another."""
    return jaro_winkler_similarity(set1, set2, p=p, max_l=max_l)
