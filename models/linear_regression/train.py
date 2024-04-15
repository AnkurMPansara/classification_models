import os
import numpy as np
import pandas as pd
import pickle as pkl

class LinearRegression:
    def __init__(self):
        self.parameters = {}

    def prepare_data(self, path, split_ratio=0.7):
        #location for training data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, path)
        #loading training data into variable
        data = pd.read_csv(data_path)
        #cleaning data
        data = data.dropna()
        # Calculate the number of rows for training and testing
        train_rows = int(len(data) * split_ratio)
        # Split the data into training and testing sets
        train_data = data.iloc[:train_rows]
        test_data = data.iloc[train_rows:]
        # training dataset and labels
        train_input = train_data['x'].values.reshape(-1, 1)
        train_output = train_data['y'].values.reshape(-1, 1)
        # validation dataset and labels
        test_input = test_data['x'].values.reshape(-1, 1)
        test_output = test_data['y'].values.reshape(-1, 1)
        #returning values
        return train_input, train_output, test_input, test_output
    
    def forward_pass(self, input):
        m = self.parameters['m']
        c = self.parameters['c']
        predictions = np.multiply(m, input) + c
        return predictions
    
    def backward_pass(self, train_input, train_output, predictions):
        derivatives = {}
        df = (predictions - train_output)
        dm = 2 * np.mean(np.multiply(train_input, df))
        dc = 2 * np.mean(df)
        derivatives['dm'] = dm 
        derivatives['dc'] = dc 
        return derivatives
    
    def cost_function(self, predictions, train_output): 
        cost = np.mean((train_output - predictions) ** 2) 
        return cost
    
    def update_parameters(self, derivatives, learning_rate): 
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm'] 
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']

    def train(self, train_input, train_output, learning_rate, iterations):
        # Initialize random parameters 
        self.parameters['m'] = np.random.uniform(0, 1) * -1
        self.parameters['c'] = np.random.uniform(0, 1) * -1
        # Initialize loss 
        self.loss = [] 
        #running loop for provided iterations
        for i in range(iterations):
            # Forward propagation 
            predictions = self.forward_pass(train_input)
            # Cost function 
            cost = self.cost_function(predictions, train_output)
            # Back propagation 
            derivatives = self.backward_pass(train_input, train_output, predictions)
            # Update parameters 
            self.update_parameters(derivatives, learning_rate) 
            # Append loss and print 
            self.loss.append(cost) 
            print("Iteration = {}, Loss = {}".format(i + 1, cost))
    
    def test(self, test_input, test_output):
        # Predict values
        predictions = self.forward_pass(test_input)
        # Threshold predictions to get binary classes
        binary_predictions = (predictions >= 0.7).astype(int)
        # Threshold test_output to get binary classes
        binary_output = (test_output >= 0.7).astype(int)
        # Calculate accuracy
        accuracy = np.mean(binary_predictions == binary_output)
        print("Accuracy = {}".format(accuracy))
    
    def save_model(self, file_path):
        #save the model into .model file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)
        with open(file_path, 'wb') as file:
            pkl.dump(self.parameters, file)
        print("Model saved in: " + file_path)

    def load_model(self, file_path):
        #load the model from .model file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)
        with open(file_path, 'rb') as file:
            self.parameters = pkl.load(file)


if __name__ == "__main__":
    # Create an instance of the LinearRegression class
    lr_model = LinearRegression()
    #preparing data
    train_input, train_output, test_input, test_output = lr_model.prepare_data('../../data/numeric_linear/data.csv', 0.7)
    #train the model
    learning_rate = 0.0001
    iterations = 20
    lr_model.train(train_input, train_output, learning_rate, iterations)
    #test the model
    lr_model.test(test_input, test_output)
    #save the model
    lr_model.save_model('output/linear_regression.model')