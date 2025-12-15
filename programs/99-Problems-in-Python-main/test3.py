# Program 2: targeted for missing mutation operators
def program2(lst):
    # 16: list comprehension
    squares = [x**2 for x in lst]

    # 17: lambda -> def
    func = lambda x: x + 1
    def func_def(x):
        return x + 1
    val = func_def(squares[0] if squares else 0)

    # 18: early return elimination
    if val > 10:
        return val

    # 22: loop unrolling
    for i in range(min(3, len(lst))):
        lst[i] += 1

    # 31, 32: bitshift & bitwise ops
    a = 4
    b = 1
    c = a << b     # bitshift left
    d = a >> b     # bitshift right
    e = a & b      # bitwise AND
    f = a | b      # bitwise OR
    g = a ^ b      # bitwise XOR

    # 33: remove not
    flag = not (val > 0)

    # 40: empty dict
    mydict = {}

    # 41: empty tuple
    mytuple = ()

    # 45: while to false
    i = 0
    while i < len(lst):
        lst[i] += 2
        i += 1

    # 49: remove raise
    try:
        if val < 0:
            raise ValueError("Negative value")
    except ValueError:
        pass

    # 53: len check to false
    if len(lst) > 0:
        total = sum(lst)
    else:
        total = 0

    # 54: len(x) == 0 -> True
    empty_list = []
    if len(empty_list) == 0:
        empty_flag = True

    # 55: len(x) > 0 -> False
    if len(lst) > 0:
        has_items = True
    else:
        has_items = False

    return squares, val, lst, c, d, e, f, g, flag, mydict, mytuple, total, empty_flag, has_items
