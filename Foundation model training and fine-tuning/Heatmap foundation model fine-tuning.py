import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models_mae import mae_vit_large_patch16
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torchvision
from collections import Counter
import torch
import numpy as np
import os

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# Function to calculate category weights
def calculate_class_weights(dataset):
    all_labels = [label for _, label in dataset.imgs]
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)
    class_weights = {label: total_samples / count for label, count in label_counts.items()}
    return class_weights


def save_predictions(dataset, model, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model.eval()

    sample_names = []
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for inputs, labels, paths in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            sample_names.extend(os.path.basename(path) for path in paths)
            true_labels.extend(labels.cpu().numpy())
            predicted_probs.extend(probs[:, 1].cpu().numpy())

    df = pd.DataFrame({
        'Sample Name': sample_names,
        'label': true_labels,
        'Predicted Probability': predicted_probs
    })

    df.to_csv(save_path, index=False)
    return save_path


# ---------------------
# Initialize the model
# ---------------------
# Initialize the basic model
base_model = mae_vit_large_patch16()

# Create linear classification head
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.linear(x)


# Create the entire model, including the base model and classification head
class CustomModel(nn.Module):
    def __init__(self, base_model, classifier):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.classifier = classifier

    def forward(self, x):
        x = self.base_model.forward_encoder(x, mask_ratio=0.0)[0]
        x = x[:, 0]
        x = self.classifier(x)
        return x


# Initialize the classifier
classifier = LinearClassifier(input_dim=1024, hidden_dim=512, output_dim=2)

# Initialize the complete model
model = CustomModel(base_model, classifier).cuda()

# Load the saved encoder parameter file
encoder_state_dict = torch.load(r"/data/zhenyuan/pretrain/save_result/encoder_checkpoint_491.pth")

# Restore the encoder state using the loaded state dictionary
print("Loading pre-trained checkpoint from:", r"/data/zhenyuan/pretrain/save_result/encoder_checkpoint_491.pth")
state_dict = model.state_dict()
checkpoint_model = encoder_state_dict

# Remove weights that do not match the current task (such as the weights of the classification head)
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# Load pre-trained model weights
msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)

# Set the model to evaluation mode
model.eval()

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def dataload(trainDataPath, testDataPath, validationDataPath):
    batch_size = 64
    train_data = ImageFolderWithPaths(
        trainDataPath,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = ImageFolderWithPaths(
        testDataPath,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    vali_data = ImageFolderWithPaths(
        validationDataPath,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    vali_loader = DataLoader(vali_data, batch_size=batch_size, shuffle=True)

    # Print the number of samples in each data set
    print(f"Number of samples in the training set: {len(train_data)}")
    print(f"Number of samples in the test set: {len(test_data)}")
    print(f"Number of samples in the validation set: {len(vali_data)}")

    return train_data, test_data, vali_data, train_loader, test_loader, vali_loader


train_dataset = '/data/zhenyuan/dataset1/pretraining_dataset3/train'
val_dataset = '/data/zhenyuan/dataset1/pretraining_dataset3/validation'
test_dataset = '/data/zhenyuan/dataset1/pretraining_dataset3/test'

train_data, test_data, vali_data, train_loader, test_loader, val_loader = dataload(train_dataset, test_dataset,
                                                                                   val_dataset)

# ---------------------
# Training model
# ---------------------
def train(model, train_loader, val_loader, optimizer, scheduler, criterion, patience=200, epochs=200):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    log = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        train_acc = 0.
        train_labels, train_predictions = [], []

        for inputs, labels, paths in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            l2_reg = 0 * torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), p=2)
            loss += l2_reg

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()
            train_loss += loss.item()

            train_labels.extend(labels.cpu().numpy())
            train_predictions.extend(outputs[:, 1].detach().cpu().numpy())

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        train_fpr, train_tpr, _ = roc_curve(train_labels, train_predictions)
        train_roc_auc = auc(train_fpr, train_tpr)

        val_loss, val_acc, val_roc_auc = evaluate(model, val_loader, criterion)

        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_roc_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_roc_auc:.4f}')
        test(model, test_loader, criterion)

        log.append([train_loss, train_acc, val_loss, val_acc])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/data/zhenyuan/save_model(outputs)/foundation_model_HM.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Stop training at epoch {epoch} due to early stopping')
                model.load_state_dict(torch.load("/data/zhenyuan/save_model(outputs)/foundation_model_HM.pth"))
                break

    log = np.array(log)
    plt.plot(log[:, 0], label='Train Loss')
    plt.plot(log[:, 2], label='Val Loss')
    plt.legend()
    plt.show()

    plt.plot(log[:, 1], label='Train Acc')
    plt.plot(log[:, 3], label='Val Acc')
    plt.legend()
    plt.show()


def evaluate(model, loader, criterion):
    model.eval()
    loss = 0.
    correct = 0
    labels_list, predictions_list = [], []

    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            labels_list.extend(labels.cpu().numpy())
            predictions_list.extend(outputs[:, 1].detach().cpu().numpy())

    loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    fpr, tpr, _ = roc_curve(labels_list, predictions_list)
    roc_auc = auc(fpr, tpr)

    return loss, acc, roc_auc


def test(model, loader, criterion):
    test_loss, test_acc, test_roc_auc = evaluate(model, loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_roc_auc:.4f}')


# ---------------------
# Set up the training environment
# ---------------------
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Calculate category weights
train_class_weights = calculate_class_weights(train_data)
class_weights = torch.tensor([train_class_weights[i] for i in range(len(train_class_weights))],
                             dtype=torch.float).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

train(model, train_loader, val_loader, optimizer, scheduler, criterion)

train_csv_path = save_predictions(train_data, model, '/home/zhenyuan/mice/results/foundation_model_HM_train_predictions.csv')
test_csv_path = save_predictions(test_data, model, '/home/zhenyuan/mice/results/foundation_HM_test_predictions.csv')
vali_csv_path = save_predictions(vali_data, model, '/home/zhenyuan/mice/results/foundation_HM_vali_predictions.csv')