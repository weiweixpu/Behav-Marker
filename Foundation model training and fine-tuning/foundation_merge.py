from models_mae import mae_vit_large_patch16
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os
from collections import Counter


# Custom dataset class, return images, labels and paths
class ImageFolderWithPaths2(torchvision.datasets.ImageFolder):
    def __init__(self, root1, root2, transform=None):
        super().__init__(root1, transform=transform)
        self.dataset2 = torchvision.datasets.ImageFolder(root2, transform=transform)
        assert len(self.samples) == len(self.dataset2.samples)

    def __getitem__(self, index):
        original_tuple1 = super(ImageFolderWithPaths2, self).__getitem__(index)
        path1 = self.imgs[index][0]
        original_tuple2 = self.dataset2[index]
        path2 = self.dataset2.imgs[index][0]
        if os.path.basename(path1) != os.path.basename(path2):
            print(
                f"Filename {os.path.basename(path1)} in first dataset doesn't match with {os.path.basename(path2)} in the second dataset.")
        assert os.path.basename(path1) == os.path.basename(path2)
        tuple_with_path = (original_tuple1, original_tuple2, path1)
        return tuple_with_path


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to calculate category weights
def calculate_class_weights(dataset):
    all_labels = [label for _, label in dataset.samples]
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)
    class_weights = {label: total_samples / count for label, count in label_counts.items()}
    return class_weights


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seed=None):
        super(LinearClassifier, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.linear(x)

# Create the entire model, including the base model and classification head
class CustomModel(nn.Module):
    def __init__(self, base_model1, base_model2, classifier):
        super(CustomModel, self).__init__()
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.classifier = classifier

    def forward(self, x1, x2):
        x1 = self.base_model1.forward_encoder(x1, mask_ratio=0.0)[0]
        x2 = self.base_model2.forward_encoder(x2, mask_ratio=0.0)[0]
        x1 = x1[:, 0]
        x2 = x2[:, 0]
        return x1, x2


# Data loading function
def dataload(trainDataPath1, trainDataPath2, testDataPath1, testDataPath2, validationDataPath1, validationDataPath2):
    batch_size = 16
    train_set = ImageFolderWithPaths2(trainDataPath1, trainDataPath2, transform=transform)
    val_set = ImageFolderWithPaths2(validationDataPath1, validationDataPath2, transform=transform)
    test_set = ImageFolderWithPaths2(testDataPath1, testDataPath2, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, train_set


def save_predictions(model, loader, file_path, device):
    model.eval()
    predictions = []
    actuals = []
    true_labels = []

    with torch.no_grad():
        for ((inputs1, labels1), (inputs2, labels2), sample_names) in loader:
            inputs1, labels1 = inputs1.to(device), labels1.to(device)
            inputs2, labels2 = inputs2.to(device), labels2.to(device)
            outputs = model(inputs1, inputs2)
            x1, x2 = outputs
            x = torch.cat((x1, x2), dim=1)
            outputs = model.module.classifier(x)
            probabilities = torch.softmax(outputs, dim=1)
            predictions.extend(probabilities.cpu().numpy())
            actuals.extend([os.path.basename(sample) for sample in sample_names])
            true_labels.extend(labels1.cpu().numpy())

    df = pd.DataFrame({
        'Sample Name': actuals,
        'True Label': true_labels,
        'Predicted Probability': [prob.tolist() for prob in predictions]
    })

    probabilities_df = pd.DataFrame(df['Predicted Probability'].tolist())
    probabilities_df.columns = [f'Class_{i}_Probability' for i in range(probabilities_df.shape[1])]
    result_df = pd.concat([df[['Sample Name', 'True Label']], probabilities_df], axis=1)
    result_df.to_csv(file_path, index=False)
    print(f'Saved predictions to {file_path}')



def train(model, classifier, train_loader, test_loader, val_loader, criterion, optimizer, device):
    best_loss = float('inf')
    prev_vali_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    log = []
    num_epochs = 20
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(num_epochs):
        train_loss, train_acc, test_loss, test_acc, val_loss, val_acc = 0., 0., 0., 0., 0., 0.
        train_samples, test_samples, val_samples = 0, 0, 0

        all_train_labels, all_train_outputs = [], []
        all_test_labels, all_test_outputs = [], []
        all_val_labels, all_val_outputs = [], []

        model.train()
        classifier.train()

        for ((inputs1, labels1), (inputs2, labels2), _) in train_loader:
            inputs1, labels1 = inputs1.to(device), labels1.to(device)
            inputs2, labels2 = inputs2.to(device), labels2.to(device)

            # Forward pass
            x1, x2 = model(inputs1, inputs2)
            x = torch.cat((x1, x2), dim=1)
            outputs = model.module.classifier(x)

            loss = criterion(outputs, labels1)
            train_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels1).sum().item()
            train_samples += inputs1.size(0)

            all_train_labels.extend(labels1.cpu().numpy())
            all_train_outputs.extend(outputs[:, 1].detach().cpu().numpy())

        scheduler.step()

        # Verification and testing phase
        model.eval()
        classifier.eval()
        with torch.no_grad():
            for ((inputs1, labels1), (inputs2, labels2), _) in test_loader:
                inputs1, labels1 = inputs1.to(device), labels1.to(device)
                inputs2, labels2 = inputs2.to(device), labels2.to(device)

                x1, x2 = model(inputs1, inputs2)
                x = torch.cat((x1, x2), dim=1)
                outputs = classifier(x)

                test_loss += criterion(outputs, labels1).item()
                test_samples += inputs1.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == labels1).sum().item()

                all_test_labels.extend(labels1.cpu().numpy())
                all_test_outputs.extend(outputs[:, 1].detach().cpu().numpy())

            for ((inputs1, labels1), (inputs2, labels2), _) in val_loader:
                inputs1, labels1 = inputs1.to(device), labels1.to(device)
                inputs2, labels2 = inputs2.to(device), labels2.to(device)

                x1, x2 = model(inputs1, inputs2)
                x = torch.cat((x1, x2), dim=1)
                outputs = classifier(x)

                val_loss += criterion(outputs, labels1).item()
                val_samples += inputs1.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_acc += (predicted == labels1).sum().item()

                all_val_labels.extend(labels1.cpu().numpy())
                all_val_outputs.extend(outputs[:, 1].detach().cpu().numpy())

        # Calculate the average loss and accuracy
        train_loss /= train_samples
        train_acc /= train_samples
        test_loss /= test_samples
        test_acc /= test_samples
        val_loss /= val_samples
        val_acc /= val_samples

        train_fpr, train_tpr, _ = roc_curve(all_train_labels, all_train_outputs)
        train_auc = auc(train_fpr, train_tpr)

        test_fpr, test_tpr, _ = roc_curve(all_test_labels, all_test_outputs)
        test_auc = auc(test_fpr, test_tpr)

        val_fpr, val_tpr, _ = roc_curve(all_val_labels, all_val_outputs)
        val_auc = auc(val_fpr, val_tpr)

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '/data/zhenyuan/save_model(outputs)/alldata_foundation_model.pth')
            epochs_no_improve = 0
        else:
            if val_loss >= prev_vali_loss:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
            if epochs_no_improve >= patience:
                print(f'Stop training at epoch {epoch} due to early stopping')
                model.load_state_dict(torch.load('/data/zhenyuan/save_model(outputs)/alldata_foundation_model.pth'))
                save_predictions(model, train_loader, '/home/zhenyuan/mice/results/alldata_foundation_train_predictions.csv', device)
                save_predictions(model, test_loader, '/home/zhenyuan/mice/results/alldata_foundation_test_predictions.csv', device)
                save_predictions(model, val_loader, '/home/zhenyuan/mice/results/alldata_foundation_vali_predictions.csv', device)
                break

        log.append([train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])
        prev_vali_loss = val_loss

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

    log = np.array(log)
    plt.plot(log[:, 0], label='Train Loss')
    plt.plot(log[:, 2], label='Val Loss')
    plt.plot(log[:, 4], label='Test Loss')
    plt.legend()
    plt.show()

    plt.plot(log[:, 1], label='Train Acc')
    plt.plot(log[:, 3], label='Val Acc')
    plt.plot(log[:, 5], label='Test Acc')
    plt.legend()
    plt.show()



# Path settings
train_dataset1 = '/data/zhenyuan/dataset1/pretraining_dataset2/train'
train_dataset2 = '/data/zhenyuan/dataset1/pretraining_dataset3/train'
val_dataset1 = '/data/zhenyuan/dataset1/pretraining_dataset2/validation'
val_dataset2 =  '/data/zhenyuan/dataset1/pretraining_dataset3/validation'
test_dataset1 = '/data/zhenyuan/dataset1/pretraining_dataset2/test'
test_dataset2 = '/data/zhenyuan/dataset1/pretraining_dataset3/test'

train_loader, val_loader, test_loader , train_set = dataload(train_dataset1, train_dataset2, test_dataset1, test_dataset2, val_dataset1, val_dataset2)

# Loading pre-trained encoder weights
base_model1 = mae_vit_large_patch16()
base_model2 = mae_vit_large_patch16()

# Load a saved encoder parameter file
encoder_state_dict1 = torch.load("/data/zhenyuan/save_model(outputs)/big_model_TM.pth")
encoder_state_dict2 = torch.load("/data/zhenyuan/save_model(outputs)/big_model_HM.pth")

# Restore the encoder state using the loaded state dictionary
state_dict1 = base_model1.state_dict()
state_dict2 = base_model2.state_dict()
checkpoint_model1 = encoder_state_dict1
checkpoint_model2 = encoder_state_dict2


# Remove weights that do not match the current task (e.g. weights of the classification head)
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model1 and checkpoint_model1[k].shape != state_dict1[k].shape:
        print(f"Removing key {k} from pretrained checkpoint 1")
        del checkpoint_model1[k]
    if k in checkpoint_model2 and checkpoint_model2[k].shape != state_dict2[k].shape:
        print(f"Removing key {k} from pretrained checkpoint 2")
        del checkpoint_model2[k]

# Loading pre-trained model weights
msg1 = base_model1.load_state_dict(checkpoint_model1, strict=False)
msg2 = base_model2.load_state_dict(checkpoint_model2, strict=False)
print("Model 1 loading msg:", msg1)
print("Model 2 loading msg:", msg2)

# Initialize the classifier
classifier = LinearClassifier(input_dim=1024*2, hidden_dim=512, output_dim=2)
classifier.apply(init_weights)

# Calculate category weights
train_class_weights = calculate_class_weights(train_set)
class_weights = torch.tensor([train_class_weights[i] for i in range(len(train_class_weights))], dtype=torch.float).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weights)

# criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel(base_model1, base_model2, classifier)
model = nn.DataParallel(model)  # Wrap the model with DataParallel
model.to(device)

# Set up the training environment
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

train(model, classifier, train_loader, test_loader, val_loader, criterion, optimizer, device)

torch.save(model.state_dict(), '/data/zhenyuan/save_model(outputs)/alldata_foundation_model.pth')

save_predictions(model, train_loader, '/home/zhenyuan/mice/results/alldata_foundation_train_predictions.csv', device)
save_predictions(model, test_loader, '/home/zhenyuan/mice/results/alldata_foundation_test_predictions.csv', device)
save_predictions(model, val_loader, '/home/zhenyuan/mice/results/alldata_foundation_vali_predictions.csv', device)
