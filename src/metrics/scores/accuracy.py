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




# [1, 5, 6, 3, 9,
#  7, 8, 10, 23, 45,
#  67, 88]

# [1, 9, 8]
# [5, 3]


# i = None
# i1 = 0
# i3 = 3
# i5 = 1
# i8 = 6
# i9 = 4

# l = len([1, 9, 8, 5, 3]) # 5
# max_l = 12

# if i < l:
#     1.0
# else:
#     1.0 - (i - l + 1) / (max_l - l + 1)


# i1 = 1.0 # 1.0
# i3 = 1.0 # 1.0
# i5 = 1.0 # 1.0
# i8 = 1.0 # 1.0 - (6 - 5 + 1) / (12 - 5 + 1) = 1.0 - 2 / 8 = 1.0 - 0.25 = 0.75
# i9 = 1.0 # 1.0
