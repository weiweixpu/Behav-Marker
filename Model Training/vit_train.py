import os
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torch
from timm import create_model
import torch.nn.init as init
import torch.nn as nn


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def restruct_model(model, num_cls):
    """
    :param model: loaded pre-trained model
    :param num_cls: number of categories to be identified
    :return: reconstructed model with pre-training
    """
    for param in model.parameters():
        param.requires_grad = True

    # Modify the classification header
    model.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.head.in_features, num_cls)
    )

    for param in model.head.parameters():
        param.requires_grad = True

    # Randomly initialize all weights of the model
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Initialize weights using normal distribution, with a mean of 0 and a standard deviation of 0.01
            init.normal_(module.weight, mean=0, std=0.01)
            if module.bias is not None:
                # Initialize the bias to a constant 0
                init.constant_(module.bias, 0)

    print(model)
    model.cuda()
    return model


def dataload(trainData, testData, validationData):
    train_data = ImageFolderWithPaths(
        trainData,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    test_data = ImageFolderWithPaths(
        testData,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

    vali_data = ImageFolderWithPaths(
        validationData,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    vali_loader = DataLoader(vali_data, batch_size=4, shuffle=True)

    # return train_data, test_data, train_loader, test_loader
    return train_data, test_data, vali_data, train_loader, test_loader, vali_loader


def save_predictions(dataset, model, save_path):
    # Create a DataLoader with a batch size of 64 and no shuffling
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model.eval() # Set the model to evaluation mode

    sample_names = []
    predicted_probs = []
    true_labels = []  # Add a list to store the real labels

    with torch.no_grad(): # Do not calculate gradients, used in the inference phase
        for inputs, labels, paths in data_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            sample_names.extend(os.path.basename(path) for path in paths)
            predicted_probs.extend(probs[:, 1].cpu().numpy())
            true_labels.extend(labels.cpu().numpy())  # Add the true labels to the list

    # Create a pandas DataFrame to store sample names, predicted probabilities, and true labels
    df = pd.DataFrame({
        'Sample Name': sample_names,
        'Predicted Probability': predicted_probs,
        'True Label': true_labels  # Add a new column to store the actual label
    })

    df.to_csv(save_path, index=False)  # Modify to save as CSV file



# def train(model, trainData, testData):
def train(model, trainData, testData, validationData):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # 每个epoch让学习率乘以0.95
    train_data, test_data, vali_data, train_loader, test_loader, vali_loader = dataload(trainData, testData, validationData)

    log = []

    patience = 5
    best_vali_loss = float('inf')
    epochs_no_improve = 0
    prev_vali_loss = float('inf')

    epoches = 20
    for epoch in range(epoches):
        train_loss = 0.
        train_acc = 0.
        test_loss = 0.
        test_acc = 0.
        vali_loss = 0.
        vali_acc = 0.

        # 假设标签为1的样本少，我们给它的权重设为2，而标签为0的样本权重设为1
        weights = [1, 5]
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        model.train()
        for i, (inputs, labels, paths) in enumerate(train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 增加L2正则化
            params = torch.cat([x.view(-1) for x in model.parameters()])
            # 计算L2正则化，只考虑权重参数
            l2_reg = 0.0005 * torch.norm(params[:-class_weights.numel()], p=2)
            loss += l2_reg

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels, paths) in enumerate(test_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == labels).sum().item()
                test_loss += loss.item()


        model.eval()
        with torch.no_grad():
            for i, (inputs, labels, paths) in enumerate(vali_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                vali_acc += (predicted == labels).sum().item()
                vali_loss += loss.item()

        train_loss /= len(train_data)
        train_acc /= len(train_data)
        test_loss /= len(test_data)
        test_acc /= len(test_data)
        vali_loss /= len(vali_data)
        vali_acc /= len(vali_data)

        print('Epoch: ', epoch, 'Train_loss: ', train_loss, 'Train acc: ', train_acc)
        print('Test_loss: ', test_loss, 'Test acc: ', test_acc)
        print('Vali_loss: ', vali_loss, 'Vali acc: ', vali_acc)
        log.append([train_loss, train_acc, test_loss, test_acc, vali_loss ,vali_acc])

        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            torch.save(model.state_dict(), '/data/zhenyuan/save_model(outputs)/random_densenet121_TM.pth')
            epochs_no_improve = 0
        else:
            if vali_loss >= prev_vali_loss:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
            if epochs_no_improve >= patience:
                print(f'Stop training at epoch {epoch} due to early stopping')
                # 加载最佳模型
                model.load_state_dict(torch.load('/data/zhenyuan/save_model(outputs)/random_densenet121_TM.pth'))
                break

        prev_vali_loss = vali_loss  # Update previous validation loss for the next epoch

    log = np.array(log)
    plt.plot(log[:, 0], label='train loss')
    plt.plot(log[:, 2], label='test loss')
    plt.plot(log[:, 4], label='vali loss')
    plt.legend()
    plt.show()

    plt.plot(log[:, 1], label='train acc')
    plt.plot(log[:, 3], label='test acc')
    plt.plot(log[:, 5], label='vali acc')
    plt.legend()
    plt.show()


    # torch.save(model.state_dict(), '/home/lzr/huzy/mice/log/model.pth')

    save_predictions(train_data, model, '/home/zhenyuan/mice/results/random_densenet121_TM_train_predictions.csv')
    save_predictions(test_data, model, '/home/zhenyuan/mice/results/random_densenet121_TM_test_predictions.csv')
    save_predictions(vali_data, model, '/home/zhenyuan/mice/results/random_densenet121_TM_vali_predictions.csv')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = create_model('vit_base_patch16_224', pretrained=True)
    num_cls = 2  # 只有两个类别
    model = restruct_model(model, num_cls)
    train_data = '/data/zhenyuan/dataset1/pretraining_dataset2/train'
    test_data = '/data/zhenyuan/dataset1/pretraining_dataset2/test'
    vali_data = '/data/zhenyuan/dataset1/pretraining_dataset2/validation'
    train(model, train_data, test_data, vali_data)