import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
TIME_STEP = 10      # Number of time steps for the RNN
INPUT_SIZE = 1      # RNN input size (number of features)
LEARNING_RATE = 0.03  # Learning rate for the optimizer
Epoch = 450       # Number of training epochs

# Generate data
steps = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)  # Generate data points from 0 to 2π
x_np = np.sin(steps)  # Input data: sine wave
# Create a more complex target function
y_np = 0.5 * np.sin(steps) + 0.3 * np.cos(2 * steps) + 0.2 * np.exp(-steps / 5)

# Plot the original data
plt.plot(steps, y_np, 'r-', label='Target (complex function)')
plt.plot(steps, x_np, 'b-', label='Input (sin)')
plt.legend(loc='best')
plt.title('Sine Wave Input vs. Complex Function Target')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # Define the RNN layer
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # Number of hidden units in the RNN
            num_layers=1,       # Single RNN layer
            batch_first=True    # The first dimension of the input & output is the batch size
        )
        # Define the output layer
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x, h_state):
        """
        Forward pass for the RNN.
        :param x: Input tensor of shape (batch, time_step, input_size)
        :param h_state: Hidden state tensor of shape (n_layers, batch, hidden_size)
        :return: Output predictions and the updated hidden state
        """
        r_out, h_state = self.rnn(x, h_state)  # RNN forward pass

        # Compute output for each time step
        outputs = [self.output_layer(r_out[:, time_step, :]) for time_step in range(r_out.size(1))]
        return torch.stack(outputs, dim=1), h_state


# Instantiate the RNN model
model = RNN()
print(model)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

# Initialize the hidden state
hidden_state = None

# Enable interactive plotting
plt.ion()
plt.figure(figsize=(40, 6))

# Training loop
for step in range(Epoch):
    # Define the time range for the current step
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)  # Input sine wave
    y_np = 0.5 * np.sin(steps) + 0.3 * np.cos(2 * steps) + 0.2 * np.exp(-steps / 5)  # Complex target function

    # Convert numpy arrays to PyTorch tensors and reshape for RNN input
    x_tensor = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y_tensor = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    # Forward pass: predict the output
    prediction, hidden_state = model(x_tensor, hidden_state)
    
    # Detach hidden state to prevent backpropagating through the entire training history
    hidden_state = hidden_state.data

    # Compute the loss
    loss = loss_function(prediction, y_tensor)

    # Backpropagation
    optimizer.zero_grad()   # Clear previous gradients
    loss.backward()         # Compute gradients
    optimizer.step()        # Update the model parameters

    # Calculate accuracy for regression: using a tolerance for close predictions
    tolerance = 0.1  # Define tolerance level for considering prediction as accurate
    predicted_values = prediction.data.numpy().flatten()
    true_values = y_np.flatten()
    accuracy = np.mean(np.abs(predicted_values - true_values) < tolerance)

    # Plot the results
    plt.cla()  # Clear the figure for updating
    plt.plot(steps, y_np.flatten(), 'r-', label='True Value')
    plt.plot(steps, predicted_values, 'b-', label='Prediction')
    plt.title(f'Step {step+1}/{Epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')
    if step == 0:
        plt.legend(loc='best')
    plt.draw()
    plt.pause(0.1)

# Disable interactive plotting and display the final plot
plt.ioff()
plt.show()
