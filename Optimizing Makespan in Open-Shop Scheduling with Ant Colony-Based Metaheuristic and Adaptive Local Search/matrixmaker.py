import numpy as np

def reorder_processing_times(processing_times, machines):
    # Convert input lists to numpy arrays for easy manipulation
    processing_times = np.array(processing_times)
    machines = np.array(machines)

    num_rows, num_cols = processing_times.shape
    reordered_times = np.zeros((num_rows, num_cols), dtype=int)

    # Iterate over each row to reorder processing times
    for i in range(num_rows):
        row_times = processing_times[i]
        row_machines = machines[i]

        # Create a list of (machine_number, processing_time) pairs
        machine_time_pairs = list(zip(row_machines, row_times))
        
        # Sort the pairs by machine_number
        sorted_pairs = sorted(machine_time_pairs, key=lambda x: x[0])
        
        # Extract the reordered times from sorted pairs
        reordered_times[i] = [pair[1] for pair in sorted_pairs]

    return reordered_times

def parse_to_list_of_lists(data):
    # Split the input string into lines
    lines = data.strip().split('\n')
    
    # Initialize an empty list to hold the result
    result = []
    
    # Process each line
    for line in lines:
        # Split the line into string numbers and convert them to integers
        numbers = list(map(int, line.split()))
        # Append the list of numbers to the result
        result.append(numbers)
    
    return result

# Example usage
data = """
 78 73 58 60 52
 33 52 92 48 46
 31 63 86 59 80
 85  2 56 92  5
 91 49 75 78 28
"""

processing_times = parse_to_list_of_lists(data)
# Example input data


machine = """
  3  5  1  4  2
  1  4  5  2  3
  5  4  1  2  3
  2  5  4  1  3
  3  4  2  5  1
"""
machines = parse_to_list_of_lists(machine)

# Get reordered processing times
reordered_times = reorder_processing_times(processing_times, machines)


def list_of_lists_to_string(lst):
    # Convert each list of integers to a space-separated string
    lines = [' '.join(map(str, sublist)) for sublist in lst]
    
    # Join all lines with newline characters
    return '\n'.join(lines)

# Example usage
data = reordered_times

formatted_string = list_of_lists_to_string(data)
print(formatted_string)