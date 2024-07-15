from statistics import mean


def get_accurancy_from_list(ranking: list[str], expected_ranking: list[str]):
    accuracy_array = list()
    l = len(expected_ranking)
    max_length = len(ranking)
    
    for v in expected_ranking:
        i = ranking.index(v)

        if i < l:
            accuracy_array.append(1.0)
        else:
            accuracy = 1.0 - (i - l + 1) / (max_length - l + 1)
            accuracy_array.append(accuracy)

    return mean(accuracy_array)
