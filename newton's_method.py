import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application, 
    convert_xor
)

# Define a custom function for natural logarithm that will print as ln(x)
class ln(sp.Function):
    @classmethod
    def eval(cls, arg):
        # Leave unevaluated to preserve the original notation.
        return
    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/self.args[0]
        else:
            raise ValueError("Invalid argument index in ln.")

# Define a custom function for logarithm base 10 that will print as log(x)
class log(sp.Function):
    @classmethod
    def eval(cls, arg):
        # Leave unevaluated so that it prints as log(x)
        return
    def fdiff(self, argindex=1):
        if argindex == 1:
            # Derivative of log base 10: 1/(x * ln(10))
            return 1/(self.args[0] * sp.log(10))
        else:
            raise ValueError("Invalid argument index in log.")

# I will use numpy functions directly in lambdify.

def newtons_method(func, dfunc, x0, tol=1e-6, max_iter=100, derivative_threshold=1e-12):
    """
    Applies Newton's Method to find a root for the function f(x).

    If a nearly zero derivative is encountered, the method prompts the user to
    enter a different starting guess, and then restarts the iteration.
    """
    guess = x0
    while True:
        for i in range(max_iter):
            f_val = func(guess)
            fprime_val = dfunc(guess)
            
            if abs(fprime_val) < derivative_threshold:
                print(f"\nError: Derivative is nearly zero (|f'(x)| = {abs(fprime_val)}) at x = {guess}.")
                new_guess_str = input("Please enter a different starting guess: ")
                try:
                    guess = float(new_guess_str)
                except ValueError:
                    print("Invalid input! Exiting.")
                    return None
                break
            x_new = guess - f_val / fprime_val
            
            if abs(x_new - guess) < tol:
                return x_new, i + 1
            
            guess = x_new
        else:
            print("Warning: Maximum iterations reached; solution may not be accurate.")
            return guess, max_iter

def main():
    # Declare the symbol.
    x = sp.symbols('x')
    
    # Get the function from user input.
    input_expr = input("Enter the function f(x): ")
    
    # Set up transformations to allow '^' for exponentiation and implicit multiplication.
    transformations = (standard_transformations + 
                       (implicit_multiplication_application,) + 
                       (convert_xor,))
    
    # Map the variable x, Euler's number e, and the logarithmic functions.
    # Note that:
    #   if the user enters ln(x), it stays as ln(x)
    #   if the user enters log(x), it will be interpreted as base-10 log.
    local_dict = {
        "x": x,
        "e": sp.E,
        "ln": ln,  # natural logarithm (will display as ln(x))
        "log": log # base-10 logarithm (will display as log(x))
    }
    
    try:
        f_expr = parse_expr(input_expr, transformations=transformations, local_dict=local_dict)
    except Exception as e:
        print("Invalid function input!")
        return
    
    # Compute the symbolic derivative f'(x)
    f_prime_expr = sp.diff(f_expr, x)
    
    # Lambdify the expressions for numerical evaluations.
    # Map our ln to np.log (natural logarithm) and log to np.log10 (base-10 logarithm)
    numerical_module = {"ln": np.log, "log": lambda arg: np.log10(arg)}
    f = sp.lambdify(x, f_expr, modules=[numerical_module, "numpy"])
    f_prime = sp.lambdify(x, f_prime_expr, modules=[numerical_module, "numpy"])
    
    print("\nYou entered:")
    print("  f(x) =", f_expr)
    print("  f'(x) =", f_prime_expr)
    
    try:
        x0 = float(input("\nEnter the initial guess: "))
    except ValueError:
        print("Invalid initial guess!")
        return
    
    try:
        tol_input = input("Enter the tolerance (default 1e-6): ")
        tol = float(tol_input) if tol_input.strip() != '' else 1e-6
    except ValueError:
        tol = 1e-6
    
    try:
        max_iter_input = input("Enter maximum iterations (default 100): ")
        max_iter = int(max_iter_input) if max_iter_input.strip() != '' else 100
    except ValueError:
        max_iter = 100
    
    result = newtons_method(f, f_prime, x0, tol, max_iter)
    
    if result is None:
        return
    
    root, iterations = result

    if isinstance(root, complex) and abs(root.imag) < 1e-10:
        root = root.real

    print(f"\nApproximated root: {root}")
    print(f"Found in {iterations} iterations.")

if __name__ == "__main__":
    main()
