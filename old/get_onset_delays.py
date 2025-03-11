#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def get_txt_files(directory):
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_files.append(filename)
    return txt_files

def extract_integers(file_path, search_string):
    integers = []
    
    with open(file_path, 'r', encoding='utf-16-le') as file:
        for line in file:
            if search_string in line:
                # Find all integers in the line
                numbers = re.findall(r'\b\d+\b', line)
                # Convert strings to integers and add to the list
                integers.extend(map(int, numbers))
    
    return integers

# Specify the directory path
directory_path = "/mnt/data_dump/pixelstress/logfiles/"

# Get the list of .txt files
txt_file_list = get_txt_files(directory_path)

delays = []

# Loop files
for i, fn in enumerate(txt_file_list):
    
    # Specify the file path
    file_path = os.path.join(directory_path, fn)
    
    # Specify the search string
    search_string = "gridOntime.OnsetDelay"
    
    # Call the function and get the list of integers
    result = extract_integers(file_path, search_string)
    
    # Build matrix
    delays.append(np.column_stack(([i] * len(result), result)))

# Stack all
delays = np.vstack(delays)

# Create the scatter plot
plt.figure(figsize=(10, 15))
plt.scatter(delays[:, 1], delays[:, 0], c='blue', alpha=0.6, s=50)
    
