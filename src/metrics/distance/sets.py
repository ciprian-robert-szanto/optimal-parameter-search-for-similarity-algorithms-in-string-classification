import numpy as np
from nltk.metrics.distance import jaccard_distance, masi_distance

from utils.preprocess import _preprocess


def _jaccard_similarity(set1, set2):
    """Distance metric comparing set-similarity."""
    return 1 - jaccard_distance(set1, set2)


def _sorensen_dice_similarity(set1, set2):
    """Distance metric comparing set-similarity."""
    intersection = len(set1.intersection(set2))
    return 2 * intersection / (len(set1) + len(set2))


def _masi_similarity(set1, set2):
    """Distance metric that takes into account
    partial agreement when multiple labels are assigned."""
    return 1 - masi_distance(set1, set2)


def _overlap_similarity(set1, set2):
    """Distance metric comparing set-overlaping."""
    intersection = len(set1.intersection(set2))
    return intersection / min(len(set1), len(set2))


def _set_similarity_controller(measure, input_string, dataset_strings):
    # Preprocess the input
    input_tokens = set(_preprocess(input_string))

    # Select the function to use
    match measure:
        case 'jaccard':
            trigger_function = _jaccard_similarity

        case 'sorensen':
            trigger_function = _sorensen_dice_similarity

        case 'masi':
            trigger_function = _masi_similarity

        case 'overlap':
            trigger_function = _overlap_similarity

        case _:
            raise ValueError(f"Unknown similarity measure: {measure}")

    # Create an array to add each similarity
    similarities = []
    for text in dataset_strings:
        # Preprocess each phrase
        tokens = set(_preprocess(text))
        # Compute similarity
        similarity_coef = trigger_function(input_tokens, tokens)
        similarities.append(similarity_coef)

    return np.array(similarities)
