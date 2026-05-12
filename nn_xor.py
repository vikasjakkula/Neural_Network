import numpy as np
import matplotlib.pyplot as pl

class NeuralNetwork():
    def __init__(self,input_size, hidden_size, output_size, learning_rate = 0.5):
        self.W1 =  np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1,hidden_size))
        self.W2 = np.random.randn(hidden_size,output_size)
        self.b2 = np.zeros((1,output_size))
        self.learning_rate = learning_rate      

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def forward(self,X):
        self.X = X
        z1 = X@self.W1 + self.b1

        a1 = self.sigmoid(z1)

        self.z1 = z1
        self.a1 = a1

        z2 = a1@self.W2 + self.b2

        a2 = self.sigmoid(z2)

        self.z2 = z2
        return a2
    
    def compute_loss(self,y_true,y_pred):
        y_diff = y_pred - y_true
        return 0.5*(np.mean(y_diff**2))
    
    def sigmoid_derivative(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def backward(self,y_true,y_pred):
        m = self.X.shape[0]

        error = y_pred - y_true
        d_output = error * self.sigmoid_derivative(self.z2)

        d_W2 = (self.a1.T @ d_output) / m
        d_b2 = np.sum(d_output, axis=0, keepdims=True) / m
        
        # Backpropagate to hidden layer
        d_hidden = (d_output @ self.W2.T) * self.sigmoid_derivative(self.z1)
        
        d_W1 = (self.X.T @ d_hidden) / m
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1


    # Test the network
if __name__ == "__main__":
    # Create a simple network: 2 inputs, 3 hidden neurons, 1 output
    nn = NeuralNetwork(2, 3, 1, learning_rate=0.5)
    
    # Create some test data (XOR problem)
    X = np.array([[0, 0],
                  [0, 1], 
                  [1, 0],
                  [1, 1]])
    
    # True labels for XOR
    y_true = np.array([[0],
                       [1],
                       [1], 
                       [0]])
    
    print("Training the neural network on XOR problem...")
    print("=" * 50)
    
    loss_his = []
    # Training loop
    epochs = 15000
    for epoch in range(epochs):
        # Forward pass
        predictions = nn.forward(X)
        
        # Calculate loss
        loss = nn.compute_loss(y_true, predictions)
        
        # Backward pass (update weights)
        nn.backward(y_true, predictions)
        loss_his.append(loss)
        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    print(f"\nFinal Epoch {epochs}, Loss: {loss:.4f}")
    print("=" * 50)
    
    # Test final predictions
    final_predictions = nn.forward(X)
    print("\nFinal Results:")
    print("Input    | Predicted | Actual | Correct?")
    print("-" * 40)
    for i in range(len(X)):
        predicted = 1 if final_predictions[i] > 0.5 else 0
        # The fix: 
        actual = int(y_true[i][0])
        correct = "✓" if predicted == actual else "✗"
        print(f"{X[i]}     |    {predicted}     |   {actual}   |   {correct}")
    
    print(f"\nFinal predictions (raw): {final_predictions.flatten()}")
    print(f"True labels: {y_true.flatten()}")

    pl.figure(figsize=(8, 5))
    pl.plot(loss_his, label='Training Loss')
    pl.xlabel('Epochs')
    pl.ylabel('Loss')
    pl.title('Neural Network Training Loss on XOR Problem')
    pl.legend()
    pl.grid(True)
    pl.show()