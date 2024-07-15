import re
import statistics
from random import random


def get_lines(filepath: str, remove_duplicates=False, write_to_cache: bool = False, cache_filepath: str = './temps/output_data.txt') -> list[str]:

    if not isinstance(filepath, str):
        return []

    with open(filepath, 'r') as read_file:
        lines: list[str] = read_file.readlines()
        if remove_duplicates:
            lines: set[str] = set(lines)
        parsed_lines: list[str] = list(map(replace_useless, lines))

        if write_to_cache == True and isinstance(cache_filepath, str):
            with open(cache_filepath, 'w') as cache_file:
                cache_file.write(str(parsed_lines))

        return parsed_lines


def replace_useless(text: str) -> str:
    result: str = re.sub(r"\n|\"", "", text, flags=re.M | re.I)
    result = ' '.join(result.split())
    result = result.lstrip()
    return result


def get_mode_length(data: list[str]):
    lengths_array = list(map(lambda x: len(x), data))
    return statistics.mode(lengths_array)


def get_random_string(data: list[str], length=None):
    temp_data = data
    if not (length is None):
        temp_data = list(filter(lambda x: len(x) == length, data))

    random_index = int(random() * (len(temp_data) + 1))
    return temp_data[random_index]


def get_strings_from_features(data: list[str], features: list[str], mcf: int = 1):
    result_array = list()
    features_set = set(list(map(lambda x : x.lower(), features)))

    for k in data:
        k_set = set(k.lower().split())
        if len(k_set.intersection(features_set)) >= mcf:
            result_array.append(k)

    return result_array