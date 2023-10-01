import csv
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Download the Iris dataset manually and save it as "iris.data" in your working directory.

# Step 2: Load and Parse the Dataset

# Initialize empty lists to store data
data = []
labels = []

# Open and read the dataset file
with open("iris.data", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) == 5:  # Ensure valid data rows
            data_row = [float(val) for val in row[:-1]]  # Convert attributes to float
            label = row[-1]
            data.append(data_row)
            labels.append(label)

# Convert data and labels to NumPy arrays for further analysis
data = np.array(data)
labels = np.array(labels)

# Print the number of features (attributes)
num_attributes = data.shape[1]
print(f"Number of attributes per sample: {num_attributes}")

# Print the different species and the number of samples for each
unique_species, species_counts = np.unique(labels, return_counts=True)
print(f"Different species: {unique_species}")
print(f"Samples per species: {species_counts}")

# Step 3: Create Scatterplots for All Pairs of Attributes

# Define colors for each species
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}

# Create scatterplots for all pairs of attributes
for i in range(num_attributes):
    for j in range(i + 1, num_attributes):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, i], data[:, j], c=[colors[label] for label in labels])
        plt.xlabel(f'Attribute {i + 1}')
        plt.ylabel(f'Attribute {j + 1}')
        plt.title(f'Scatterplot of Attribute {i + 1} vs. Attribute {j + 1}')
        plt.show()
