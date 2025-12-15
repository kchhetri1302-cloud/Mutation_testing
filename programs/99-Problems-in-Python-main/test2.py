# Program 2: functions, control flow, exception, break/continue
def program2(lst):
    total = 0
    try:
        for val in lst:
            if val > 0:                    # relational_op
                total += val
            else:
                continue                   # remove_continue
            if val == 5:
                break                      # remove_break
    except Exception:
        pass                               # remove_except_body

    if total == 0:                         # len_eq_zero_to_true
        total = 1                           # return_none_to_one

    def inner_func(x):                      # lambda_to_def
        return x + 1

    result = inner_func(total)

    if total > 0:                           # len_gt_zero_to_false / if_true_to_false
        return result                       # return_to_none
    else:
        return None                         # return_none_to_one
