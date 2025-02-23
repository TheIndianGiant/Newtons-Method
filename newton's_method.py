import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application, 
    convert_xor
)

def newtons_method(func, dfunc, x0, tol=1e-6, max_iter=100):
    """
    Applies Newton's Method to find a root for the function f(x).

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
               Maximum number of iterations.

    Returns:
      (root, iterations) if converged, or (approx_root, max_iter) if maximum iterations reached.
    """
    x = x0
    for i in range(max_iter):
        f_val = func(x)
        fprime_val = dfunc(x)
        
        if fprime_val == 0:
            print("Error: Zero derivative encountered. No solution found.")
            return None
        
        x_new = x - f_val / fprime_val
        
        # Check if the update is within the specified tolerance.
        if abs(x_new - x) < tol:
            return x_new, i + 1
        
        x = x_new
    print("Warning: Maximum iterations reached; solution may not be accurate.")
    return x, max_iter

def main():
    # Declare the symbol we will use.
    x = sp.symbols('x')
    
    # Get the function from user input.
    input_expr = input("Enter the function f(x): ")
    
    # Set up transformations to allow '^' for exponentiation and implicit multiplication.
    transformations = (standard_transformations + 
                       (implicit_multiplication_application,) + 
                       (convert_xor,))
    try:
        f_expr = parse_expr(input_expr, transformations=transformations)
    except Exception as e:
        print("Invalid function input!")
        return
    
    # Compute the symbolic derivative f'(x).
    f_prime_expr = sp.diff(f_expr, x)
    
    # Lambdify expressions for numerical evaluations.
    f = sp.lambdify(x, f_expr, 'numpy')
    f_prime = sp.lambdify(x, f_prime_expr, 'numpy')
    
    print("\nYou entered:")
    print("  f(x) =", f_expr)
    print("  f'(x) =", f_prime_expr)
    
    # Prompt the user to enter the initial guess.
    try:
        x0 = float(input("\nEnter the initial guess: "))
    except ValueError:
        print("Invalid initial guess!")
        return
    
    # Allow the user to input the tolerance and maximum iterations.
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

    # If the result has a negligible imaginary part, display only the real part.
    # Here, I consider an imaginary part less than 1e-10 as negligible.
    if isinstance(root, complex) and abs(root.imag) < 1e-10:
        root = root.real

    print(f"\nApproximated root: {root}")
    print(f"Found in {iterations} iterations.")

if __name__ == "__main__":
    main()
