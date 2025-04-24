import pandas as pd
from datetime import datetime, timedelta
import os
import subprocess
import sys

# Function to read CSV data and return a DataFrame
def read_data(filename: str) -> pd.DataFrame:
    print(f"Reading data from file: {filename}")  # Debug output
    with open(filename) as f:
        lines = f.readlines()

    start_date = ""
    start_time = ""

    # Find start date and time from the metadata
    for line in lines:
        if line.startswith("Start Time"):
            start_time = line.split()[-1].strip().strip('"')  # Strip quotes
        elif line.startswith("Start Date"):
            start_date = line.split()[-1].strip().strip('"')  # Strip quotes
        if line.startswith("Accelerometer X"):
            break

    # Convert start date and time to a datetime object
    start = datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M:%S")

    # Read the accelerometer data using pandas
    data = pd.read_csv(filename, skiprows=len(lines) - len(lines[lines.index(line):]), header=0)

    # Create timestamps based on start time and 80 Hz sample rate
    step = timedelta(seconds=1 / 80)
    timestamps = [start + i * step for i in range(len(data))]

    # Create DataFrame with timestamps
    df_data = pd.DataFrame(data.values, columns=data.columns)
    df_data['time'] = timestamps
    df_data = df_data[['time'] + list(data.columns)]

    return df_data

# Function to process data and save as CSV
def data_to_csv(input_files: list, output_directory: str) -> list:
    output_csv_paths = []  # Initialize the list to track output CSV paths
    
    for file in input_files:
        try:
            df = read_data(file)  # Read data from the file
            base_name = os.path.basename(file)  # Get the base name of the file
            output_csv_filename = f"{os.path.splitext(base_name)[0]}.csv"  # CSV filename
            output_csv_path = os.path.join(output_directory, output_csv_filename)  # Full path
            
            # Write to CSV
            df.to_csv(output_csv_path, index=False)
            print(f"Processed and saved to {output_csv_path}")  # Confirmation output
            output_csv_paths.append(output_csv_path)  # Add path to the list

        except Exception as e:
            print(f"Failed to process {file}: {e}")  # Catch errors for specific files

    return output_csv_paths  # Return paths of processed CSVs

# Function to save CSV paths to a text file
def save_csv_paths_to_textfile(output_csv_paths: list, textfile_path: str):
    with open(textfile_path, 'w') as f:
        for csv_path in output_csv_paths:
            f.write(f'stepcount "{csv_path}" --txyz time,"Accelerometer X","Accelerometer Y","Accelerometer Z"\n')
        f.write(":END\n")  # Mark the end of the file
    #print(textfile_path)

def run_anaconda_script(textfile_path: str, env_name: str):
    # Create a temporary batch script
    batch_filename = "run_conda_commands.bat"
    
    # Construct the commands to be written into the batch file
    with open(batch_filename, 'w') as batch_file:
        batch_file.write(f"@echo off\n")  # Disable echoing of commands
        batch_file.write(f"call conda activate {env_name}\n")  # Activate the Conda environment
        batch_file.write(f"cmd /c \"type {textfile_path}\"\n")  # Use 'type' to output content for debug
        batch_file.write(f"cmd < {textfile_path}\n")  # Execute commands from the text file
        batch_file.write("pause\n")  # Optional: Keep command prompt open after execution
    
    # Run the batch file in a new command prompt window
    subprocess.run(f'start cmd /k "{batch_filename}"', shell=True)

# Main function to execute the workflow
def main(address_text_file: str, output_directory: str):
    if not os.path.isfile(address_text_file):
        print("The provided address file does not exist. Exiting.")
        return

    with open(address_text_file, 'r') as f:
        input_files = [line.strip().strip('"') for line in f if line.strip()]

    if not input_files:
        print("No valid input files found in the provided address file. Exiting.")
        return

    if not os.path.isdir(output_directory):
        print("Provided output directory does not exist. Exiting.")
        return

    # Process data and save to separate CSV files
    output_csv_paths = data_to_csv(input_files, output_directory)

    # Save the generated paths to a text file
    textfile_path = os.path.join(output_directory, "output_paths.txt")
    save_csv_paths_to_textfile(output_csv_paths, textfile_path)

    # Run the Anaconda script after processing
    run_anaconda_script(textfile_path, "stepcount")

# Entry point for script execution
if __name__ == "__main__":
    if len(sys.argv) != 3:  # Ensure expected argument count
        print("Usage: python pipeline_oxford_stepcount.py <address_text_file> <output_directory>")
        sys.exit(1)

    address_text_file = sys.argv[1]  # First argument
    output_directory = sys.argv[2]    # Second argument

    main(address_text_file, output_directory)  # Call main with the parsed arguments

# python pipeline_oxford_stepcount.py DS_10_address.txt DS_10_test