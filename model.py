import torchvision
from torch import nn
from torch.nn import functional as F


def build_model(num_classes, for_train=True):
    # model = Pokeball(num_classes=num_classes)
    model = torchvision.models.resnet18(pretrained=for_train)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class Pokeball(nn.Module):
    def __init__(self, num_classes):
        super(Pokeball, self).__init__()
        self.seq = nn.Sequential(
            # 定义第一个卷积层
            # in_channels=3 表示输入的图像有3个颜色通道 (RGB)
            # out_channels=32 表示将输出32个特征映射（即卷积核数量）
            # kernel_size=3 表示卷积核的大小是3x3

            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            # 定义第二个卷积层，接收来自第一个卷积层的32个特征映射作为输入
            nn.Conv2d(16, 32, kernel_size=5, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            # # 定义第二个卷积层，接收来自第一个卷积层的32个特征映射作为输入
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            #
            # # 定义池化层
            # # kernel_size=2 表示在2x2的区域上进行最大池化
            #
            # # # 定义第三个卷积层
            # nn.Conv2d(64, 128, kernel_size=5, padding=1),
            # nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2),
            #
            # nn.Conv2d(128, 256, kernel_size=5, padding=1),
            # nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2),
            #
            # nn.Conv2d(256, 512, kernel_size=5, padding=1),
            # nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            # nn.Linear(512, 4096),
            # nn.ReLU(),

            nn.Dropout(),
            # 定义一个全连接/线性层
            # in_features=4096 是根据前面卷积层的输出特征数量确定的
            # 例如，如果卷积后的特征图大小是16x16，且有128个过滤器，则总特征数是 16*16*128
            # out_features=num_classes 代表了最终的输出类别数
            nn.Linear(93312, num_classes)
        )

    def forward(self, x):
        return self.seq(x)
