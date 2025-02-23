import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application, 
    convert_xor
)

# Custom numerical log function that distinguishes the base.
def my_log(x, base=None):
    """
    Evaluate the logarithm of x.
    
    If base is None, computes the natural logarithm.
    If base equals 10, computes the base-10 logarithm.
    Otherwise, computes logarithm with the given base.
    """
    if base is None:
        return np.log(x)
    elif base == 10:
        return np.log10(x)
    else:
        return np.log(x) / np.log(base)

def newtons_method(func, dfunc, x0, tol=1e-6, max_iter=100, derivative_threshold=1e-12):
    """
    Applies Newton's Method to find a root for the function f(x).

    If a nearly zero derivative is encountered, the method prompts the user to
    enter a different starting guess, and then restarts the iteration.

    Parameters:
      func : callable
             The function f(x) for which to find the root.
      dfunc: callable
             The derivative of f(x).
      x0   : float
             The initial guess.
      tol  : float
             The tolerance for convergence.
      max_iter: int
               Maximum number of iterations per attempt.
      derivative_threshold: float
               Threshold below which the derivative is considered nearly zero.
    
    Returns:
      (root, iterations) if converged, or (approx_root, max_iter) if maximum iterations reached.
    """
    guess = x0
    # Outer loop: Allows re-attempt with a new starting guess if derivative is near zero.
    while True:
        for i in range(max_iter):
            f_val = func(guess)
            fprime_val = dfunc(guess)
            
            # Check if the derivative is nearly zero.
            if abs(fprime_val) < derivative_threshold:
                print(f"\nError: Derivative is nearly zero (|f'(x)| = {abs(fprime_val)}) at x = {guess}.")
                new_guess_str = input("Please enter a different starting guess: ")
                try:
                    guess = float(new_guess_str)
                except ValueError:
                    print("Invalid input! Exiting.")
                    return None
                # Break out of the for loop and restart with the new guess.
                break
            x_new = guess - f_val / fprime_val
            
            # Check for convergence.
            if abs(x_new - guess) < tol:
                return x_new, i + 1
            
            guess = x_new
        else:
            # The for loop completed without encountering a nearly zero derivative.
            print("Warning: Maximum iterations reached; solution may not be accurate.")
            return guess, max_iter

def main():
    # Declare the symbol
    x = sp.symbols('x')
    
    # Get the function from user input.
    input_expr = input("Enter the function f(x): ")
    
    # Set up transformations to allow '^' for exponentiation and implicit multiplication.
    transformations = (standard_transformations + 
                       (implicit_multiplication_application,) + 
                       (convert_xor,))
    
    # Map the variable x, Euler's number e, and the logarithmic functions.
    # Here, I interpret:
    #   log(x) as log base 10 and ln(x) as the natural logarithm (log base e).
    local_dict = {
        "x": x,
        "e": sp.E,
        "log": lambda arg: sp.log(arg, 10),  # log(x) becomes log base 10
        "ln": sp.log                       # ln(x) remains the natural logarithm
    }
    
    try:
        f_expr = parse_expr(input_expr, transformations=transformations, local_dict=local_dict)
    except Exception as e:
        print("Invalid function input!")
        return
    
    # Compute the symbolic derivative f'(x)
    f_prime_expr = sp.diff(f_expr, x)
    
    # Lambdify the expressions for numerical evaluations using a custom mapping for log.
    # Note: The key "log" must be a string.
    f = sp.lambdify(x, f_expr, modules=[{"log": my_log}, "numpy"])
    f_prime = sp.lambdify(x, f_prime_expr, modules=[{"log": my_log}, "numpy"])
    
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

    # If the result is complex with a negligible imaginary part, display only the real part.
    if isinstance(root, complex) and abs(root.imag) < 1e-10:
        root = root.real

    print(f"\nApproximated root: {root}")
    print(f"Found in {iterations} iterations.")

if __name__ == "__main__":
    main()
