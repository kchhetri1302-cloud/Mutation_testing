def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def is_even(n):
    return n % 2 == 0

def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


# Add this at the bottom of the file
if __name__ == "__main__":
    import sys
    import ast
    
    # Read input from stdin
    input_data = sys.stdin.read().strip()
    
    if not input_data:
        sys.exit(0)
    
    try:
        # Try to parse as Python literal (for lists, tuples, etc.)
        args = ast.literal_eval(input_data)
    except:
        # If that fails, split by whitespace
        args = input_data.split()
    
    # Test different functions based on input
    if len(args) == 1:
        n = int(args[0]) if isinstance(args, list) else int(args)
        print(f"is_even({n}) = {is_even(n)}")
        print(f"factorial({n}) = {factorial(n)}")
    elif len(args) == 2:
        a = int(args[0]) if isinstance(args, list) else int(args[0])
        b = int(args[1]) if isinstance(args, list) else int(args[1])
        print(f"add({a}, {b}) = {add(a, b)}")
        print(f"subtract({a}, {b}) = {subtract(a, b)}")
        print(f"multiply({a}, {b}) = {multiply(a, b)}")
        try:
            print(f"divide({a}, {b}) = {divide(a, b)}")
        except ValueError as e:
            print(f"divide({a}, {b}) = ERROR: {e}")
    else:
        print("Please provide 1 or 2 numbers as input")