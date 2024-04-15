import os
import numpy as np
import pandas as pd
import pickle as pkl

class LogisticRegression:
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
        w = self.parameters['w']
        b = self.parameters['b']
        x = np.multiply(w, input) + b
        predictions =  1 / (1 + np.exp(-x))
        return predictions
    
    def backward_pass(self, train_input, train_output, predictions):
        derivatives = {}
        N = train_input.shape[0]
        dL = (1/N) * np.sum(-train_output/predictions + (1 - train_output)/(1 - predictions))
        dy = np.multiply(predictions, (1 - predictions))
        dw = (1/N) * np.dot((dL * dy).T, train_input)
        db = (1/N) * np.sum(dL * dy)
        derivatives['dw'] = dw
        derivatives['db'] = db 
        return derivatives
    
    def cost_function(self, predictions, train_output): 
        cost = -np.mean(train_output * np.log(predictions) + (1 - train_output) * np.log(1 - predictions))
        return cost
    
    def update_parameters(self, derivatives, learning_rate): 
        self.parameters['w'] = self.parameters['w'] - learning_rate * derivatives['dw'] 
        self.parameters['b'] = self.parameters['b'] - learning_rate * derivatives['db']

    def train(self, train_input, train_output, learning_rate, iterations):
        # Initialize random parameters 
        self.parameters['w'] = np.random.randn(1) * 0.01
        self.parameters['b'] = 0
        # Initialize loss 
        self.loss = []
        prev_cost = float('inf') 
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
        #predict values
        predictions = self.forward_pass(test_input)
        #calculate its accuracy
        error = np.abs(predictions - test_output) / np.abs(test_output)
        accuracy = np.mean(1 - error)
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
    # Create an instance of the LogisticRegression class
    lg_model = LogisticRegression()
    #preparing data
    train_input, train_output, test_input, test_output = lg_model.prepare_data('../../data/numeric_probabilistic/data.csv', 0.7)
    #train the model
    learning_rate = 0.001
    iterations = 1000
    lg_model.train(train_input, train_output, learning_rate, iterations)
    #test the model
    lg_model.test(test_input, test_output)
    #save the model
    lg_model.save_model('output/linear_regression.model')