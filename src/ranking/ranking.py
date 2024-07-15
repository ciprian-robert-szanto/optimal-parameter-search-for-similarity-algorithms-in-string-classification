import numpy as np

from metrics.distance.compute_similarity import compute_weighted_similarities


def rank_strings(weights, input_string, dataset_strings, max_rows=None):
    """
    Calculate distance and rank strings.

    The weights will be applied as w1 / sum(w).
    Weights [0, 100]:
        - Vectors
            - cosine
            - fasttext

        - Sets
            - jaccard
            - sorensen
            - masi
            - overlap

        - Sequences
            - jaro
            - jaro_winkler

        - Matrix
            - original_levenshtein
            - improved_levenshtein


    ### Cosine
    Effective in capturing the semantic similarity between documents.

    Use a high weight for cosine similarity when dealing with long texts or documents
    where the context and frequency of terms matter significantly.


    ### FastText


    ### Jaccard
    Effective for binary or categorical data.

    Use a moderate weight when the presence or absence of
    terms (rather than their frequency) is more important.

    Useful for short texts or cases where you
    want to measure the overlap of unique terms.


    ### Sorensen
    Effective for being sensitive to common elements.

    Use a moderate weight when you need a balance between the size of
    the overlap and the size of the sets.

    Useful for short to medium-length texts


    ### Masi


    ### Jaro


    ### Jaro Winkler


    ### Overlap
    Effective when comparing sets of different sizes.

    Useful when dealing with lists or short texts where partial overlap is significant.


    ### Original Levenshtein

    ### Improved Levenshtein

    """
    weighted_similarities = compute_weighted_similarities(
        weights, input_string, dataset_strings)

    sorted_indices = np.argsort(weighted_similarities)[::-1]
    ranked_strings = [dataset_strings[i] for i in sorted_indices]

    if not (max_rows is None) and max_rows > 0:
        print(f"Ranked strings based on weighted similarity to the input string:")
        for idx, string in enumerate(ranked_strings):
            if max_rows is None or idx < max_rows:
                print(
                    f"{idx + 1}: {string} (Score: {weighted_similarities[sorted_indices[idx]]})")
        
    return ranked_strings
