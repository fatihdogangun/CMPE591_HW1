import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
    act = torch.tensor(action[i], dtype=torch.int32).squeeze()
    action_onehot = F.one_hot(act.long(), num_classes=4).float()
    samples.append({
        'image': torch.tensor(image[i] / 255, dtype=torch.float32),
        'action': action_onehot,
        'position': torch.tensor(position[i], dtype=torch.float32),
        'image_after': torch.tensor(image_after[i] / 255, dtype=torch.float32)
    })


train_set, temp = train_test_split(samples, test_size=0.3, random_state=42)
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(validation, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.bottleneck_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((512 + 4) * 4 * 4, 512 * 4 * 4), 
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4))
        )
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), 
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), 
            nn.ReLU()
        )
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), 
            nn.ReLU()
        )
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),  
            nn.ReLU()
        )
        
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)    
        self.dec5_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid() 
        )
    
    def forward(self, x, action):
      
        e1 = self.enc1(x)    
        e2 = self.enc2(e1)   
        e3 = self.enc3(e2)  
        e4 = self.enc4(e3)   
        e5 = self.enc5(e4)   
        
        action_expanded = action.view(action.size(0), action.size(1), 1, 1).expand(-1, -1, e5.size(2), e5.size(3))
        bottleneck_input = torch.cat([e5, action_expanded], dim=1)  
        b = self.bottleneck_fc(bottleneck_input) 
        
      
        d1 = self.up1(b)     
        d1 = torch.cat([d1, e4], dim=1)  
        d1 = self.dec1_conv(d1)          
        
        d2 = self.up2(d1)   
        d2 = torch.cat([d2, e3], dim=1)  
        d2 = self.dec2_conv(d2)          
        
        d3 = self.up3(d2)   
        d3 = torch.cat([d3, e2], dim=1)  
        d3 = self.dec3_conv(d3)         
        
        d4 = self.up4(d3)   
        d4 = torch.cat([d4, e1], dim=1) 
        d4 = self.dec4_conv(d4)         
        
        d5 = self.up5(d4)   
        d5 = self.dec5_conv(d5)       
        
        out = self.final_conv(d5) 
        return out
    

def train(model = UNet(), train_loader = train_dataloader, valid_loader = valid_dataloader, num_epochs=1500, lr=1e-3, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    train_losses = []
    valid_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            actions = batch["action"].to(device)
            targets = batch["image_after"].to(device)


            
            optimizer.zero_grad()
            outputs = model(images, actions)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch["image"].to(device)
                actions = batch["action"].to(device)
                targets = batch["image_after"].to(device)

                outputs = model(images, actions)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * images.size(0)
        valid_loss = running_val_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {valid_loss:.6f}")
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), os.path.join("save", "hw1_3.pth"))
        

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("UNet Loss Plot")
    plt.legend()
    plt.savefig("UNet Loss Plot.png")
    plt.close()
    
 
def test(device = None, num_images = 3):
    
    model = UNet()
    model.load_state_dict(torch.load(os.path.join("save", "hw1_3.pth")))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    dataloader = test_dataloader

    batch = next(iter(dataloader))
    images = batch["image"]
    actions = batch["action"]
    images_after = batch["image_after"]

    
    with torch.no_grad():
        preds = model(images, actions)
    
    images = images.cpu().numpy()
    images_after = images_after.cpu().numpy()
    preds = preds.cpu().numpy()

    num_images = min(num_images, images.shape[0])
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4*num_images))
    if num_images == 1:
        axes = [axes]
    for i in range(num_images):
        
        ax = axes[i][0]
        ax.imshow(np.transpose(images[i], (1, 2, 0)))
        ax.set_title("Input (Before)")
        ax.axis("off")
        
        ax = axes[i][1]
        ax.imshow(np.transpose(images_after[i], (1, 2, 0)))
        ax.set_title("Ground Truth (After)")
        ax.axis("off")
        
        ax = axes[i][2]
        ax.imshow(np.transpose(preds[i], (1, 2, 0)))
        ax.set_title("Predicted (After)")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("UNet Predictions.png")
    plt.close()


#train()
test()