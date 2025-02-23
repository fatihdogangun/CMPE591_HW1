import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

load_path = "save"
samples = []
num_samples = 1000

image = np.load(os.path.join(load_path,"image.npy"))
action = np.load(os.path.join(load_path,"action.npy"))
position =np.load(os.path.join(load_path,"position.npy"))
image_after = np.load(os.path.join(load_path,"image_after.npy"))

for i in range(num_samples):
    action[i] = torch.tensor(action[i], dtype=torch.int32)
    action_onehot = F.one_hot(torch.tensor(action[i], dtype=torch.long), num_classes=4).float()
    samples.append({
        'image': torch.tensor(image[i] / 255, dtype=torch.float32),
        'action': action_onehot,
        'position': torch.tensor(position[i], dtype=torch.float32),
        'image_after': torch.tensor(image_after[i] / 255, dtype=torch.float32)
    })

train_set, temp = train_test_split(samples, test_size=0.3, random_state=42)
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_size = 3 * 128 * 128 + 4 

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2) 
        )

    def forward(self, image, action):
        
        batch_size = image.shape[0]
        flat_image = image.view(batch_size, -1)
        action = action.view(batch_size, -1)
        x = torch.cat([flat_image, action], dim=1)
        output = self.model(x)  
        return output


def train(model = MLP(), train_dataloader = train_dataloader, validation_dataloader = validation_dataloader, num_epochs=300, lr=1e-3, device=None):
    
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_loss_list = []
    validation_loss_list = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for batch in train_dataloader:
            imgs = batch["image"].to(device)
            actions = batch["action"].to(device)
            positions = batch["position"].to(device)

            optimizer.zero_grad()
            outputs = model(imgs, actions)

            loss = criterion(outputs, positions)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * imgs.shape[0]

        epoch_loss = running_train_loss / len(train_dataloader.dataset)
        train_loss_list.append(epoch_loss) 

        model.eval()
        running_val_loss = 0.0
        for batch in validation_dataloader:
            imgs = batch["image"].to(device)
            actions = batch["action"].to(device)
            positions = batch["position"].to(device)

            outputs = model(imgs, actions)
            loss = criterion(outputs, positions)
            running_val_loss += loss.item() * imgs.shape[0]

        validation_epoch_loss = running_val_loss / len(validation_dataloader.dataset)
        validation_loss_list.append(validation_epoch_loss)

        if validation_epoch_loss < best_loss:
            best_loss = validation_epoch_loss
            torch.save(model.state_dict(), os.path.join("save", "hw1_1.pth"))

        print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {epoch_loss:.6f} | Validation Loss: {validation_epoch_loss:.6f}")
        
    title = "MLP Loss Plot"
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(validation_loss_list, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close()

def test(test_dataloader = test_dataloader, device=None):
    
    model = MLP()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
   
    model.load_state_dict(torch.load(os.path.join("save", "hw1_1.pth")))

    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in test_dataloader:
            imgs = batch["image"].to(device)
            actions = batch["action"].to(device)
            positions = batch["position"].to(device)

            outputs = model(imgs, actions)
            loss = criterion(outputs, positions)
            print(f"Test Loss: {loss.item()}")



train()
test()