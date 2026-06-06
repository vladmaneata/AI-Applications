import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. The kitchen blueprint (hyperparameters) ---
num_ingredients = 4  # flour, sugar, temp, humidity
staff_per_station = 16  # neurons in hidden layer
learning_rate = 0.001  # how cautiously the chef learns
epochs = 2000  # total training iterations

# --- 2. Building the lab (The MLP architecture) ---
class CakeLab(nn.Module):
    def __init__(self):
        super(CakeLab, self).__init__()
        # Station 1: The mixer (input to hidden)
        self.mixer = nn.Linear(num_ingredients, staff_per_station)
        # Chef's logic: ReLU (If total signal < 0, output 0)
        self.activation = nn.ReLU()
        # Station 2: The oven (hidden to output score)
        self.oven = nn.Linear(staff_per_station, 1)

    def forward(self, x):
        # Forward pass: The baking sequence
        x = self.mixer(x)
        x = self.activation(x)
        cake_score = self.oven(x)
        return cake_score

# Initialize the lab
model = CakeLab()

# --- 3. The data (Past cakes / Training examples) ---
# Normalize the data (dividing by 100)
ingredients = torch.tensor([
        [5.0, 2.0, 1.8, 0.45],
        [4.5, 1.5, 1.7, 0.40],
        [5.5, 2.5, 1.9, 0.50],
        [4.8, 1.8, 1.8, 0.42]
], dtype=torch.float32)

# Scores given by the critic (Target)
true_scores = torch.tensor([[85.0], [70.0], [95.0], [80.0]], dtype=torch.float32)

# --- 4. The executive chef (Loss & Optimizer) ---
critic = nn.MSELoss()  # Mean Squared Error (unhappiness metric)
# The optimizer manages the weights and biases (the dials)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 5. Training(The practice sessions) ---
print("Chef is starting to practice...\n")

for epoch in range(epochs):
    # A. Forward Pass
    predictions = model(ingredients)
    # B. Calculate Loss
    loss = critic(predictions, true_scores)
    # C. Backpropagation (Feedback)
    optimizer.zero_grad()  # Reset feedback from last cake
    loss.backward()  # Calculate how to turn the dials
    # D. Update (Turn the dials)
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Session {epoch + 1}: Critic's unhappiness = {loss.item():.4f}")

# --- 6. Prediction (Baking a new cake) ---
print("\n--- Training complete ---")
# Ingredient input: [Flour: 510g, Sugar: 210g, Temp: 182C, Humid: 48%]
new_cake = torch.tensor([[5.1, 2.1, 1.82, 0.48]], dtype=torch.float32)

model.eval()  # Tell the chefs we are now in the final exam
with torch.no_grad():
    prediction = model(new_cake)
    print(f"Predicted score for the new cake: {prediction.item():.2f}/100")