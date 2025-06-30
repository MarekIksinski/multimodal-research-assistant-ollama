#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 01:15:00 2025
@author: marek (w/ Gemini)

This module provides a statistics tool that can be called by the LLM agent.
It performs various statistical calculations on a list of numbers.
"""
import statistics
from typing import List, Union, Dict, Any

def _calculate_full_summary(data: List[Union[int, float]]) -> str:
    """Helper function to compute a full statistical summary."""
    if len(data) < 2:
        return "Cannot compute full summary: at least two data points are required."
    
    mean = statistics.mean(data)
    median = statistics.median(data)
    try:
        mode = statistics.mode(data)
    except statistics.StatisticsError:
        mode = "No unique mode"
    
    stdev = statistics.stdev(data)
    variance = statistics.variance(data)
    
    return (
        f"Statistical Summary:\n"
        f"- Count: {len(data)}\n"
        f"- Mean (Average): {mean:.4f}\n"
        f"- Median: {median:.4f}\n"
        f"- Mode: {mode}\n"
        f"- Standard Deviation: {stdev:.4f}\n"
        f"- Variance: {variance:.4f}\n"
        f"- Min: {min(data)}\n"
        f"- Max: {max(data)}"
    )

def execute_statistics_tool(operation: str, data: List[Union[int, float]]) -> str:
    """
    The main entry point for the statistics tool.

    Args:
        operation (str): The statistical operation to perform.
        data (List): A list of numbers (integers or floats).

    Returns:
        str: The result of the calculation or an error message.
    """
    # --- Data Validation ---
    if not isinstance(data, list) or not data:
        return "Error: 'data' must be a non-empty list of numbers."
    if not all(isinstance(x, (int, float)) for x in data):
        return "Error: All items in 'data' list must be numbers (integers or floats)."

    # --- Operation Routing ---
    try:
        if operation == "mean":
            result = statistics.mean(data)
            return f"The mean (average) is: {result:.4f}"
        elif operation == "median":
            result = statistics.median(data)
            return f"The median is: {result:.4f}"
        elif operation == "mode":
            result = statistics.mode(data)
            return f"The mode is: {result}"
        elif operation == "stdev":
            if len(data) < 2: return "Error: Standard deviation requires at least two data points."
            result = statistics.stdev(data)
            return f"The standard deviation is: {result:.4f}"
        elif operation == "variance":
            if len(data) < 2: return "Error: Variance requires at least two data points."
            result = statistics.variance(data)
            return f"The variance is: {result:.4f}"
        elif operation == "full_summary":
            return _calculate_full_summary(data)
        else:
            return f"Error: Unknown operation '{operation}'. Available operations: mean, median, mode, stdev, variance, full_summary."

    except statistics.StatisticsError as e:
        return f"Calculation Error: {e}. For example, mode requires a unique most common value."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


if __name__ == '__main__':
    print("--- Running Statistics Tool Demo ---")
    sample_data = [10, 15, 15, 22, 18, 25, 30, 15]
    print(f"Sample Data: {sample_data}\n")

    print("Testing 'mean'...")
    print("Result:", execute_statistics_tool("mean", data=sample_data))
    print("-" * 20)

    print("Testing 'median'...")
    print("Result:", execute_statistics_tool("median", data=sample_data))
    print("-" * 20)

    print("Testing 'mode'...")
    print("Result:", execute_statistics_tool("mode", data=sample_data))
    print("-" * 20)
    
    print("Testing 'stdev'...")
    print("Result:", execute_statistics_tool("stdev", data=sample_data))
    print("-" * 20)

    print("Testing 'full_summary'...")
    print("Result:", execute_statistics_tool("full_summary", data=sample_data))
    print("-" * 20)
    
    print("Testing error handling...")
    print("Result:", execute_statistics_tool("mean", data=[10, 'a', 20]))
    print("-" * 20)