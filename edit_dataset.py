import csv
import random
import os

def combine_and_shuffle_csv(input_folder, output_file):
    # Get the names of all CSV files in the folder
    file_names = os.listdir(input_folder)
    file_names = [f for f in file_names if f.endswith(".csv")]

    # Read and combine the CSV files
    combined_lines = []
    for file_name in file_names:
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            lines = list(reader)
            combined_lines.extend(lines)

    # Shuffle the combined lines
    random.shuffle(combined_lines)

    # Write the shuffled lines to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(combined_lines)

    print("File successfully saved: ", output_file)

# Example usage
input_folder = 'csv_files_folder'  # Folder containing the CSV files
output_file = 'combined_and_shuffled.csv'  # Output file name

combine_and_shuffle_csv(input_folder, output_file)
