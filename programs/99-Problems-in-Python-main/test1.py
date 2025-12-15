# Program 1: collection, loops, arithmetic, logic
def program1(x):
    myList = []
    for i in range(x):
        myList.append(i + 0)          # arithmetic_add_zero
        myList.append(0 + i)          # arithmetic_zero_add
        myList.append(i * 1)          # arithmetic_mul_one
        myList.append(1 * i)          # arithmetic_one_mul
        myList.append(i / 1)          # arithmetic_div_one
        myList.append(i ** 1)         # arithmetic_pow_one
        myList.append(+i)             # arithmetic_unary_pos
        myList.append(--i)            # arithmetic_double_neg

    if x > 0 and x < 10:              # logical_op_replacement
        val = True                     # bool_constant_swap
    else:
        val = False

    if x in [1,2,3]:                  # in_op_replacement
        val2 = True
    if x is not None:                  # is_op_replacement
        val3 = True

    return myList, val, val2, val3
