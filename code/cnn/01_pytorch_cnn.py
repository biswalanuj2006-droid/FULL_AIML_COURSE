# ============================================================
# PYTORCH — MNIST CNN (template for any image model)
# Run: python 01_pytorch_cnn.py
# Requires: torch, torchvision
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------------------------------------------------
# 0. Device: CUDA -> MPS -> CPU
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu")
print("Device:", device)

# ------------------------------------------------------------
# 1. Data + transforms
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean/std
])

train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

# ------------------------------------------------------------
# 2. Model — conv -> relu -> pool -> conv -> pool -> fc
# ------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()          # includes softmax

# ------------------------------------------------------------
# 3. Training loop (one epoch)
# ------------------------------------------------------------
def train_one_epoch():
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(train_data)


# ------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate():
    model.eval()
    correct = total = 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += len(yb)
    return correct / total


for epoch in range(5):
    loss = train_one_epoch()
    acc = evaluate()
    print(f"epoch {epoch + 1}: loss={loss:.4f}  test_acc={acc:.4f}")

# ------------------------------------------------------------
# 5. Save + reload + inference
# ------------------------------------------------------------
torch.save(model.state_dict(), "cnn_mnist.pt")
model2 = CNN().to(device)
model2.load_state_dict(torch.load("cnn_mnist.pt", map_location=device))
model2.eval()

x_sample, _ = next(iter(test_loader))
with torch.no_grad():
    pred = model2(x_sample[:1].to(device)).argmax(dim=1).item()
print("Sample prediction:", pred)

# ------------------------------------------------------------
# Golden rules:
#  1. model.train()/model.eval() switch dropout/batchnorm behavior.
#  2. Wrap inference in @torch.no_grad().
#  3. Save only state_dict, not the whole model.
#  4. Move BOTH inputs and model to the same device.
# ============================================================
