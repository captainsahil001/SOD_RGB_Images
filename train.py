
from models.u2net import U2NET
from utils import SaliencyDataset
from torch.utils.data import DataLoader
import torch, os
from tqdm import tqdm


device = torch.device('cpu')


model = U2NET().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()


dataset = SaliencyDataset('datasets/images', 'datasets/masks')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


os.makedirs('outputs/checkpoints', exist_ok=True)


for epoch in range(1, 11): 
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch}"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch} Loss: {epoch_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), f"outputs/checkpoints/u2net_epoch{epoch}.pth")
