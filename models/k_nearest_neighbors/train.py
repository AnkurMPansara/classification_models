import os
import numpy as np
import pandas as pd
import pickle as pkl

class KNN:
    def __init__(self):
        self.parameters = {}
    
    def prepare_data(self, path, split_ratio=0.7):
        # Location for training data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, path)
        # Loading training data into variable
        data = pd.read_csv(data_path)
        # Cleaning data
        data = data.dropna()
        # Calculate the number of rows for training and testing
        train_rows = int(len(data) * split_ratio)
        # Split the data into training and testing sets
        train_data = data.iloc[:train_rows]
        test_data = data.iloc[train_rows:]
        # Training dataset and labels
        train_input = train_data.drop(columns=['y']).values
        train_output = train_data['y'].values.reshape(-1, 1)
        # Validation dataset and labels
        test_input = test_data.drop(columns=['y']).values
        test_output = test_data['y'].values.reshape(-1, 1)
        # Returning values
        return train_input, train_output, test_input, test_output
    
    def get_normal_factors(self, input_data):
        # Convert dataset to NumPy array for faster computation
        dataset_array = np.array(input_data)
        # Calculate the minimum and maximum values for each column
        min_values = np.min(dataset_array, axis=0)
        max_values = np.max(dataset_array, axis=0)
        # Combine the minimum and maximum values into a list of tuples
        minmax = np.array([(min_val, max_val) for min_val, max_val in zip(min_values, max_values)])
        return minmax
    
    def normalize_data(self, input_data, minmax):
        # Calculate the range for each feature
        ranges = minmax[:, 1] - minmax[:, 0]
        # Normalize each feature in the dataset
        normalized_dataset = (input_data - minmax[:, 0]) / ranges
        return normalized_dataset.tolist()
    
    def get_k_nearest_neighbors(self, input_point, train_input, train_output, k):
        # Convert input_point and train_input to NumPy arrays
        input_point = np.array(input_point)
        train_input = np.array(train_input)
        # Calculate distances between the input point and all points in the dataset
        distances = np.sqrt(np.sum((train_input - input_point)**2, axis=1))
        # Get indices of K nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        # Get the K nearest neighbors
        k_nearest_neighbors = train_output[nearest_indices]
        return k_nearest_neighbors
    
    def classify_point(self, k_nearest_neighbors):
        # Count occurrences of each unique label
        unique_labels, label_counts = np.unique(k_nearest_neighbors, return_counts=True)
        # Find labels with the highest frequency (mode)
        max_count = np.max(label_counts)
        most_frequent_labels = unique_labels[label_counts == max_count]
        # Randomly select one of the most frequent labels
        random_label = np.random.choice(most_frequent_labels)
        return random_label
    
    def test(self, train_input, train_output, test_input, test_output, k):
        correct_predictions = 0
        total_predictions = len(test_input)
        # Normalize training input data
        train_minmax = self.get_normal_factors(train_input)
        normalized_train_input = self.normalize_data(train_input, train_minmax)
        # Normalize test input data using the same normalization factors as training data
        normalized_test_input = self.normalize_data(test_input, train_minmax)
        # Predict class for every test input data point
        for i in range(total_predictions):
            # Get the ith normalized test data point
            test_point = normalized_test_input[i]
            # Find its K nearest neighbors in the normalized training data
            k_nearest_neighbors = self.get_k_nearest_neighbors(test_point, normalized_train_input, train_output, k)
            # Classify the test data point
            predicted_label = self.classify_point(k_nearest_neighbors)
            # Get the actual label of the test data point
            actual_label = test_output[i]
            
            # Check if the predicted label matches the actual label
            if predicted_label == actual_label:
                correct_predictions += 1
        
        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy


if __name__ == '__main__':
    # Create an instance of the KNN class
    knn = KNN()
    # Load and split the data
    train_input, train_output, test_input, test_output = knn.prepare_data('../../data/classification_euclidean/data.csv', 0.7)
    # Define the value of k for KNN
    k = 3  # You can change this value as needed
    # Evaluate the accuracy of the KNN algorithm on the test set
    accuracy = knn.test(train_input, train_output, test_input, test_output, k)
    print("Accuracy of the KNN algorithm:", accuracy)