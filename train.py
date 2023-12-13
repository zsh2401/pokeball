import os.path
import random

import torch
import torchvision.models
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import PokemonTrainDataset, PokemonTestDataset
from model import Pokeball, build_model
import matplotlib.pyplot as plt

from test import do_test

batch_size = 64
seed = 3407

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_dataset = PokemonTrainDataset()
test_dataset = PokemonTestDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 步骤4: 定义模型
num_classes = train_dataset.classes

model = build_model(num_classes)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量 {total_params}")

# 步骤6: 训练模型

b = 0
bs = []
losses = []
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.set_title('Loss Curve')

if device == "cuda" and torch.cuda.device_count() > 1:
    print("Launch parallel computing")
    model = nn.DataParallel(model)

model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
epoch = 0
accuracy = 0

if os.path.isfile("pokeball.pth"):
    checkpoint = torch.load("pokeball.pth")
    if "accuracy" not in checkpoint:
        checkpoint["accuracy"] = 0
    accuracy = checkpoint['accuracy']
    print(f"Resume checkpoint trained {checkpoint['epoch']}, accuracy is {(accuracy * 100):.2f}%")
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

while True:
    i = 0
    for images, labels,_,_ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        # optimizer.
        optimizer.step()

        b += 1
        bs.append(b)
        losses.append(loss.item())
        i += 1
        print(f'Epoch [{epoch + 1}] [{(batch_size * i / len(train_dataset)) * 100:.2f}%], Loss: {loss.item()}')

    epoch += 1
    scheduler.step()
    print("Testing...")
    accuracy = do_test(device, model.state_dict(), epochs=epoch, test_dataset=test_dataset)

    print("Saving state")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": accuracy,
    }, "pokeball.pth")
    print("Saved state")

ax1.plot(bs, losses)
plt.show()
plt.savefig("./loss.png")
torch.save(model.state_dict(), "./pokeball.pth")
