def program2_full(lst):
    # 18: early return elimination
    if lst:
        return lst[0]

    # 22: loop unrolling
    for i in range(3):  # fixed small loop
        lst.append(i)

    # 53: len check to false
    if len(lst):  # should trigger len_check_to_false
        total = sum(lst)
    else:
        total = 0

    return lst, total
