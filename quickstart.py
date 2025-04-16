import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Only normalization for test data
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download training data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=train_transform,
)

# Download test data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=test_transform,
)

# Split training data into train and validation
train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

# Create data loaders
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Print sample data shapes
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),            # Wider first layer
            nn.BatchNorm1d(1024),              # Add batch normalization
            nn.ReLU(),
            nn.Dropout(0.3),                   # Add dropout for regularization
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    correct = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Track accuracy
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    # Return average loss and accuracy
    return total_loss / len(dataloader), correct / size

# Validation/test function
def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return accuracy

# Training loop with early stopping
def train_with_early_stopping(epochs, patience=5):
    best_accuracy = 0
    no_improvement = 0
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        print(f"Train: Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f}")
        
        # Validate
        val_accuracy = evaluate(val_dataloader, model, loss_fn)
        
        # Update learning rate based on validation accuracy
        scheduler.step(1 - val_accuracy)  # Using accuracy as the metric (higher is better)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model improved, saved checkpoint.")
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs.")
            if no_improvement >= patience:
                print(f"Early stopping at epoch {t+1}")
                break
                
    print(f"Best validation accuracy: {best_accuracy*100:.2f}%")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    return model

# Train the model with early stopping
epochs = 20
final_model = train_with_early_stopping(epochs, patience=5)

# Evaluate on test set
print("Evaluating on test set:")
test_accuracy = evaluate(test_dataloader, final_model, loss_fn)
print(f"Final test accuracy: {test_accuracy*100:.2f}%")

print("Done!")

# Class names for reference
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Function to make predictions on a single sample
def predict(model, x):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted_class = pred.argmax(1).item()
    return predicted_class, classes[predicted_class]

# sample = next(iter(test_dataloader))[0][0]
# class_index, class_name = predict(model, sample.unsqueeze(0))
# print(f"Predicted class: {class_name} ({class_index})")