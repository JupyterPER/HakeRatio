def runtime(func_str, r=3, n=1000, output=True):
    """
    Measures the runtime of a given function string using IPython's %timeit.

    Parameters:
    func_str (str): The function as a string to be timed.
    r (int): The number of repetitions.
    n (int): The number of loops per repetition.
    output (bool): If True, prints the timing result; otherwise, suppresses output.

    Returns:
    The timing result from %timeit; either printed or returned as a string.
    """
    import io
    import contextlib

    # Format the magic command for timeit with the given repetitions and loops.
    magic_command = f'-r{r} -n{n} -o {func_str}'  # Added '-o' to return the result object

    if output:
        # Directly return the output if printing is required
        return get_ipython().run_line_magic('timeit', magic_command)
    else:
        # Suppress printed output and return the result
        with contextlib.redirect_stdout(io.StringIO()):
            result = get_ipython().run_line_magic('timeit', magic_command)
        return result

import pandas as pd

def show_colwidth(df, col_width = 150):
    with pd.option_context('display.max_colwidth', col_width):
        display(df)

def show_allrowscols(df, fullcolwidth=False, col_width=150):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        if fullcolwidth:
            show_colwidth(df, col_width)
        else:
            display(df) 