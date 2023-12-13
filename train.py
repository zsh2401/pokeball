import os.path
import random

import torch
import torchvision.models
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import PokemonTrainDataset, PokemonTestDataset, PokemonValidDataset
from model import Pokeball, build_model
import matplotlib.pyplot as plt

from test import do_test



batch_size = 64
if "BATCH_SIZE" in os.environ:
    batch_size = int(os.environ["BATCH_SIZE"])

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
validation_dataset = PokemonValidDataset()
test_dataset = PokemonTestDataset()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

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

    ## 训练 ##
    for images, labels, _, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        b += 1
        bs.append(b)
        losses.append(loss.item())
        i += 1
        print(f'Epoch [{epoch + 1}] [{(batch_size * i / len(train_dataset)) * 100:.2f}%], Loss: {loss.item()}')

    epoch += 1
    scheduler.step()

    print("Validating")

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, image_paths, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的最大logit值索引作为预测结果

            # print(predicted)
            # for i, p in enumerate(predicted):
            #     print(f"Model predict {image_paths[i]} is {test_dataset.label_code_to_text(p)}")

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'***Validation Set Accuracy: {accuracy * 100:.2f}% ***')

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
