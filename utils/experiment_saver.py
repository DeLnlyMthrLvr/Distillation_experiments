import os
import csv

import os
import csv

def save_experiment_results(file_path, headers, data):
    """
    Save experiment results to a CSV file, creating the file and parent directories if necessary.

    Args:
        file_path (str): Path to the CSV file.
        headers (list): List of column headers.
        data (list): List of data to write to the CSV file.
    """

    # Create the full save path with model name
    file_path = os.path.join(file_path, "experiment_results.csv") 
    
    try:
        # Ensure headers and data are both lists and have the same length
        if not isinstance(headers, list) or not isinstance(data, list):
            raise ValueError("Headers and data must be lists.")
        if len(headers) != len(data):
            raise ValueError("Headers and data must have the same length.")

        # Create the parent directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")

        # Check if the file exists
        file_exists = os.path.isfile(file_path)

        # Open the file in append mode ('a' for append)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # If the file does not exist, write the headers first
            if not file_exists:
                writer.writerow(headers)

            # Write the data row
            writer.writerow(data)

        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        print(f"Error while saving data to {file_path}: {e}")
