import numpy as np
from numpy.typing import ArrayLike


def product_dataset(
    dataset: ArrayLike,
    range: ArrayLike,
    column_number: int,
    return_original_column=False,
):
    """
    This helper method receives a dataset (a 2d array)
    and will produce a form of combinatorial product of the dataset and the range (a 1d array).
    It will return a 2d array that has `len(range) * len(dataset)` rows.

    Specifically, it will produce a dataset where each original row is repeated `len(range)` times,
    and the column at `column_number` takes every possible value from the range.
    """
    # Create a dataset where each row is repeated |range| times.
    repeated_rows = np.repeat(dataset, len(range), axis=0)
    # Create a new column that repeats the full range vector as many times
    # as there were rows in the original data.
    new_column = np.tile(range, len(dataset))
    # Replace column.
    original_column = np.copy(repeated_rows[:, column_number])
    repeated_rows[:, column_number] = new_column

    if return_original_column:
        return repeated_rows, original_column

    return repeated_rows

def product_dataset_modified(
    dataset,
    range1,
    column_number1,
    range2,
    column_number2,
    return_original_columns=False,
):
    total_repetitions = len(range1) * len(range2)
    repeated_rows = np.repeat(dataset, total_repetitions, axis=0)
    
    new_column1 = np.tile(np.repeat(range1, len(range2)), len(dataset))
    new_column2 = np.tile(np.tile(range2, len(range1)), len(dataset))
    
    if return_original_columns:
        original_column1 = np.copy(repeated_rows[:, column_number1])
        original_column2 = np.copy(repeated_rows[:, column_number2])
        repeated_rows[:, column_number1] = new_column1
        repeated_rows[:, column_number2] = new_column2
        return repeated_rows, (original_column1, original_column2)
    else:
        repeated_rows[:, column_number1] = new_column1
        repeated_rows[:, column_number2] = new_column2
        return repeated_rows
    
def permute_features(dataset, column_numbers):
    permuted_dataset = np.copy(dataset)
    for column_number in column_numbers:
        np.random.shuffle(permuted_dataset[:, column_number])
    return permuted_dataset

def add_noise_to_features(dataset, column_numbers, noise_level=0.1):
    noisy_dataset = np.copy(dataset).astype(np.float64)  
    for column_number in column_numbers:
        noise = np.random.normal(loc=0.0, scale=noise_level, size=(dataset.shape[0],))
        noisy_dataset[:, column_number] -= noise
    return noisy_dataset