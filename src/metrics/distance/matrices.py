import re

import numpy as np
from nltk.metrics import edit_distance


def _levenshtein(measure: str, input_string: str, dataset_strings: list[str], lowercase: bool = True):
    """
    Distance metric using levenshtein.

    Since the algorithm returns the edit distance,
    the function will return a value between {0, 1} depending
    on the max edit distance.
    """
    match measure:
        case 'original_levenshtein':
            trigger_function = originalLevenshtein

        case 'improved_levenshtein':
            trigger_function = improvedLevenshtein

        case _:
            raise ValueError(f"Unknown similarity measure: {measure}")

    similarities = np.array([trigger_function(input_string, text, lowercase=lowercase)
                            for text in dataset_strings])
    return 1 - similarities / similarities.max()


def originalLevenshtein(firstString: str, secondString: str, lowercase: bool = False):
    if lowercase:
        firstString, secondString = firstString.lower(), secondString.lower()
    return edit_distance(firstString, secondString)


def improvedLevenshtein(firstPhrase: str, secondPhrase: str, lowercase: bool = False):
    splitTemplate = ' '

    # /\s+/g -> find each contiguous string of space
    firstWords = re.sub(
        r'/\s+/g', ' ', firstPhrase).strip().split(splitTemplate)
    secondWords = re.sub(
        r'/\s+/g', ' ', secondPhrase).strip().split(splitTemplate)

    levenshteinMinSum = 0

    for i in range(len(firstWords)):
        firstWord = firstWords[i]

        minLevenshteinWord = 0

        for j in range(len(secondWords)):
            secondWord = secondWords[j]

            levValue = originalLevenshtein(
                firstWord, secondWord, lowercase=lowercase)

            if (j == 0):
                minLevenshteinWord = levValue

            elif levValue < minLevenshteinWord:
                minLevenshteinWord = levValue

        levenshteinMinSum += minLevenshteinWord

    return levenshteinMinSum
