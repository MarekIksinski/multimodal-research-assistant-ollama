#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 8 17:15:00 2025

@author: marek (w/ Gemini) - CORRECTED VERSION

This module provides a robust, safe calculator tool powered by the SymPy library.
This version correctly handles trigonometric functions with inputs in degrees.
"""
from sympy import sympify, diff, expand, solve, SympifyError, pi
# Import the standard sympy trig functions with a different name
from sympy import sin as sympy_sin, cos as sympy_cos, tan as sympy_tan
from sympy.abc import x, y, z # Common symbols
import traceback

# --- NEW: Define custom trig functions that work in degrees ---
def sin_degrees(angle):
    """Calculates sine of an angle given in degrees."""
    return sympy_sin(angle * pi / 180)

def cos_degrees(angle):
    """Calculates cosine of an angle given in degrees."""
    return sympy_cos(angle * pi / 180)

def tan_degrees(angle):
    """Calculates tangent of an angle given in degrees."""
    return sympy_tan(angle * pi / 180)

# --- NEW: Create a local namespace for sympify to use ---
# This makes the tool automatically use our degree-based functions
# when it sees "sin", "cos", or "tan" in the expression string.
_local_namespace = {
    "sin": sin_degrees,
    "cos": cos_degrees,
    "tan": tan_degrees,
    "pi": pi
}

# A dictionary mapping operation names to their respective functions
_operations = {}

def _register_op(name):
    """A decorator to register a function in the operations dictionary."""
    def decorator(func):
        _operations[name] = func
        return func
    return decorator

@_register_op("evaluate")
def _evaluate_expression(expression: str) -> str:
    """Safely evaluates a numerical or symbolic expression, handling trig in degrees."""
    try:
        # MODIFIED: Use the `locals` argument to pass our custom trig functions
        expr = sympify(expression, locals=_local_namespace)
        result = expr.evalf()
        return str(result)
    except (SympifyError, TypeError, ValueError) as e:
        return f"Error: Invalid expression for evaluation. Details: {e}"
    except Exception:
        return f"An unexpected error occurred during evaluation: {traceback.format_exc()}"

@_register_op("differentiate")
def _differentiate_expression(expression: str, symbol: str) -> str:
    """Differentiates an expression with respect to a symbol."""
    try:
        expr = sympify(expression)
        sym = sympify(symbol)
        derivative = diff(expr, sym)
        return str(derivative)
    except (SympifyError, TypeError, ValueError) as e:
        return f"Error: Invalid expression or symbol for differentiation. Details: {e}"
    except Exception:
        return f"An unexpected error occurred during differentiation: {traceback.format_exc()}"

@_register_op("expand_polynomial")
def _expand_polynomial(expression: str) -> str:
    """Expands a polynomial expression."""
    try:
        expr = sympify(expression)
        expanded_expr = expand(expr)
        return str(expanded_expr)
    except (SympifyError, TypeError, ValueError) as e:
        return f"Error: Invalid expression for expansion. Details: {e}"
    except Exception:
        return f"An unexpected error occurred during expansion: {traceback.format_exc()}"

@_register_op("solve_equation")
def _solve_equation(equation: str, symbol: str) -> str:
    """Solves an equation for a given symbol."""
    try:
        expr = sympify(equation)
        sym = sympify(symbol)
        solution = solve(expr, sym)
        return str(solution)
    except (SympifyError, TypeError, ValueError) as e:
        return f"Error: Invalid equation or symbol for solving. Details: {e}"
    except Exception:
        return f"An unexpected error occurred during solving: {traceback.format_exc()}"


def execute_calculator_tool(operation: str, **kwargs) -> str:
    """The main entry point for the calculator tool."""
    if operation in _operations:
        func = _operations[operation]
        try:
            return func(**kwargs)
        except TypeError as e:
            return f"Error: Incorrect arguments provided for operation '{operation}'. Details: {e}"
    else:
        return f"Error: Unknown operation '{operation}'. Available operations: {list(_operations.keys())}"

#if __name__ == '__main__':
#    print("--- Running Calculator Tool Demo (Corrected for Degrees) ---")#
#
#    # 1. Test the skyscraper problem
#    expr1 = "50 * tan(82) + 1"
#    print(f"Evaluating '{expr1}' (expecting ~356.8)...")
#    print("Result:", execute_calculator_tool("evaluate", expression=expr1))
#    print("-" * 20)

 #   # 2. Differentiation (unchanged)
#    expr2 = "x**3 + 2*x**2 + 5"
#    print(f"Differentiating '{expr2}' with respect to 'x'...")
#    print("Result:", execute_calculator_tool("differentiate", expression=expr2, symbol="x"))
#    print("-" * 20)