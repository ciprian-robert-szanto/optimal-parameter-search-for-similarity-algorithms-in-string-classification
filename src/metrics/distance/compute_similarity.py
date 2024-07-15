import numpy as np

from metrics.distance.matrices import _levenshtein
from metrics.distance.sequences import _sequence_similarity_controller
from metrics.distance.sets import _set_similarity_controller
from metrics.distance.vectors import _vector_similarity_controller


def compute_similarity(measure, input_string, dataset_strings):
    """Function to compute similarity distance based on the selected measure"""

    match measure:
        case 'tfidf_cosine' | 'tfidf_euclidean' | 'fasttext_cosine':
            similarities = _vector_similarity_controller(
                measure, input_string, dataset_strings)

        case 'jaccard' | 'sorensen' | 'masi' | 'overlap':
            similarities = _set_similarity_controller(
                measure, input_string, dataset_strings)

        case 'jaro' | 'jaro_winkler':
            similarities = _sequence_similarity_controller(
                measure, input_string, dataset_strings)

        case 'original_levenshtein' | 'improved_levenshtein':
            similarities = _levenshtein(measure, input_string, dataset_strings)

        case _:
            raise ValueError(f"Unknown similarity measure: {measure}")

    return similarities


def compute_weighted_similarities(weights, input_string, dataset_strings):
    """Function to compute weighted similarity distance."""

    total_similarity = np.zeros(len(dataset_strings))
    total_weight = sum(weights.values())

    for measure, weight in weights.items():
        if weight > 0.0:
            similarities = compute_similarity(
                measure, input_string, dataset_strings)
            total_similarity += weight * similarities

    return total_similarity / total_weight
