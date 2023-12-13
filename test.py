import torch
from torch.utils.data import DataLoader

from dataset import PokemonTestDataset
from model import Pokeball, build_model


def do_test(device, state_dict=None, epochs=0, test_dataset=PokemonTestDataset()):
    test_loader = test_dataset.to_loader()
    # 加载你的模型结构和权重
    model = build_model(test_dataset.classes)
    if torch.cuda.device_count() > 1:
        print("Launch CUDA parallel testing")
        model = torch.nn.DataParallel(model)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        checkpoint = torch.load('./pokeball.pth')
        print(f"Loaded {checkpoint['epoch']} model from file.")
        model.load_state_dict(checkpoint["model_state_dict"])  # 加载权重

    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 正确的预测数和总的预测数
    correct = 0
    total = 0

    with torch.no_grad():  # 在这个块中，不跟踪梯度，节省内存和计算
        for images, labels, image_paths, text_labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的最大logit值索引作为预测结果
            # print(predicted)
            # for i, p in enumerate(predicted):
                # print(f"{image_paths[i]} is {test_dataset.label_code_to_text(p)}")

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 打印准确率
    accuracy = correct / total
    print(f'*** Accuracy: {accuracy * 100:.2f}% ***')

    return accuracy


if __name__ == "__main__":
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    do_test(device)
