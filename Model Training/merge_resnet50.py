import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import pandas as pd
import os

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
        assert os.path.basename(path1) == os.path.basename(path2)  # Make sure the file names of the two dataset paths are the same
        tuple_with_path = (original_tuple1, original_tuple2, path1)
        return tuple_with_path

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


class FinalModel(nn.Module):
    def __init__(self, model1, model2):
        super(FinalModel, self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(2000, 2)  # Output size for binary classification is 2

    def forward(self, x1, x2):
        # Get features from the first model
        x1 = self.model1(x1)
        # Get features from the second model
        x2 = self.model2(x2)

        # Concatenate the features along dim=1
        x = torch.cat((x1, x2), dim=1)

        # Linear classification layer
        x = self.fc(x)
        return x


def save_predictions(model, loader, file_path, device):
    model.eval()  # Set model to evaluation mode
    predictions = []
    actuals = []
    true_labels = []

    with torch.no_grad():  # Inference mode, no gradients needed
        for ((inputs1, labels1), (inputs2, labels2), sample_names) in loader:
            inputs1, labels1 = inputs1.to(device), labels1.to(device)
            inputs2, labels2 = inputs2.to(device), labels2.to(device)

            # Forward pass through the model to get raw predictions
            outputs1 = model.model1(inputs1)
            outputs2 = model.model2(inputs2)
            outputs_concat = torch.cat((outputs1, outputs2), dim=1)
            outputs = model.fc(outputs_concat)

            probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            predictions.extend(probabilities.cpu().numpy())  # Store probabilities
            actuals.extend([os.path.basename(sample) for sample in sample_names])  # Assume the third item in the loader tuple is the sample name
            true_labels.extend(labels1.cpu().numpy())  # Store true labels

    # Save to CSV file
    # Convert lists to DataFrame
    df = pd.DataFrame({
        'Sample Name': actuals,
        'True Label': true_labels,
        'Predicted Probability': [prob.tolist() for prob in predictions]  # Convert numpy arrays to lists
    })

    # Expand the 'Predicted Probability' column into multiple columns, one for each class probability
    probabilities_df = pd.DataFrame(df['Predicted Probability'].tolist())
    probabilities_df.columns = [f'Class_{i}_Probability' for i in range(probabilities_df.shape[1])]

    # Concatenate the original DataFrame (without 'Predicted Probability') with the expanded prediction probabilities
    result_df = pd.concat([df[['Sample Name', 'True Label']], probabilities_df], axis=1)

    result_df.to_csv(file_path, index=False)
    print(f'Saved predictions to {file_path}')



def train(model, train_loader, test_loader, val_loader, criterion, optimizer, device):
    best_loss = float('inf')
    prev_vali_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    log = []
    num_epochs = 5


    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(num_epochs):

        train_loss = 0.
        train_acc = 0.
        test_loss = 0.
        test_acc = 0.
        val_loss = 0.
        val_acc = 0.

        train_samples = 0
        test_samples = 0
        val_samples = 0

        model.train()
        optimizer.zero_grad()  # Clear gradients once per epoch

        # Traverse the training data loader, loading a batch of data each time
        for ((inputs1, labels1), (inputs2, labels2), _) in train_loader:
            inputs1, labels1 = inputs1.to(device), labels1.to(device) # Move the input data and labels to the specified device
            inputs2, labels2 = inputs2.to(device), labels2.to(device)
            outputs = model(inputs1, inputs2)  # Input data into the model for prediction and use the concatenated feature vector for classification
            loss = criterion(outputs, labels1)
            train_loss += loss.item()   # Accumulate the training loss of the current batch
            loss.backward()   # Back propagation to calculate gradients
            optimizer.step()   # Update model parameters based on gradient
            optimizer.zero_grad()   # Clear the gradient and prepare to process the next batch of data
            _, predicted = torch.max(outputs.data, 1)  # Calculate the training accuracy based on the prediction results and the true labels
            train_acc += (predicted == labels1).sum().item()
            train_samples += inputs1.size(0)  # Record the number of samples processed
        scheduler.step() # Update the status of the learning rate scheduler and perform operations such as learning rate decay


        model.eval()
        for ((inputs1, labels1), (inputs2, labels2), _) in test_loader:
            inputs1, labels1 = inputs1.to(device), labels1.to(device)
            inputs2, labels2 = inputs2.to(device), labels2.to(device)
            outputs1 = model.model1(inputs1)
            outputs2 = model.model2(inputs2)
            outputs_concat = torch.cat((outputs1, outputs2), dim=1)
            outputs = model.fc(outputs_concat)
            test_samples += inputs1.size(0)

            test_loss = criterion(outputs, labels1)
            _, predicted = torch.max(outputs.data, 1)
            test_acc += (predicted == labels1).sum().item()

        with torch.no_grad():
            # for ((inputs1, labels1, _), (inputs2, labels2, _)) in val_loader:
            for ((inputs1, labels1), (inputs2, labels2), _) in val_loader:
                inputs1, labels1 = inputs1.to(device), labels1.to(device)
                inputs2, labels2 = inputs2.to(device), labels2.to(device)
                outputs1 = model.model1(inputs1)
                outputs2 = model.model2(inputs2)
                outputs_concat = torch.cat((outputs1, outputs2), dim=1)
                outputs = model.fc(outputs_concat)
                val_samples += inputs1.size(0)

                val_loss = criterion(outputs, labels1)
                _, predicted = torch.max(outputs.data, 1)
                val_acc += (predicted == labels1).sum().item()

        train_loss /= train_samples
        train_acc /= len(train_loader.dataset)
        test_loss /= test_samples
        test_acc /= test_samples
        val_loss /= val_samples
        val_acc /= val_samples

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '/data/zhenyuan/save_model(outputs)/alldata_random_resnet50_merged_best_model.pth')
            epochs_no_improve = 0
        else:
            if val_loss >= prev_vali_loss:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
            if epochs_no_improve >= patience:
                print(f'Stop training at epoch {epoch} due to early stopping')
                # Load the best model
                model.load_state_dict(torch.load('/data/zhenyuan/save_model(outputs)/alldata_random_resnet50_merged_best_model.pth'))
                save_predictions(model, train_loader,
                                 '/home/zhenyuan/mice/results/alldata_resnet50_merged_train_predictions.csv', device)
                # Save the validation set sample name and predicted probability
                save_predictions(model, val_loader,
                                 '/home/zhenyuan/mice/results/alldata_resnet50_merged_val_predictions.csv', device)
                # Save the test set sample name and predicted probability
                save_predictions(model, test_loader,
                                 '/home/zhenyuan/mice/results/alldata_resnet50_merged_test_predictions.csv', device)
                break

        prev_vali_loss = val_loss  # Update previous validation loss for the next epoch

        print('Epoch: ', epoch, 'Train_loss: ', train_loss, 'Train acc: ', train_acc)
        print('Test_loss: ', test_loss.item(), 'Test acc: ', test_acc)
        print('Vali_loss: ', val_loss.item(), 'Vali acc: ', val_acc)
        log.append([train_loss, train_acc, test_loss.item(), test_acc, val_loss.item(),val_acc])

        # Append logs after each epoch
        log.append([train_loss, train_acc, test_loss.item(), test_acc, val_loss.item(), val_acc])

    # Plot the logs
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

    return model

def main():
    # Set the device and dataset path
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') #if torch.cuda.is_available() else 'cpu'
    train_data1 = '/data/zhenyuan/dataset1/pretraining_dataset2/train'
    train_data2 = '/data/zhenyuan/dataset1/pretraining_dataset3/train'
    test_data1 = '/data/zhenyuan/dataset1/pretraining_dataset2/test'
    test_data2 = '/data/zhenyuan/dataset1/pretraining_dataset3/test'
    val_data1 = '/data/zhenyuan/dataset1/pretraining_dataset2/validation'
    val_data2 = '/data/zhenyuan/dataset1/pretraining_dataset3/validation'

    batch_size = 4

    # Load the dataset
    train_set = ImageFolderWithPaths2(train_data1, train_data2, transform=transform)
    val_set = ImageFolderWithPaths2(val_data1, val_data2, transform=transform)
    test_set = ImageFolderWithPaths2(test_data1, test_data2, transform=transform)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)

    # Create two independent resnet50 models
    model1 = torchvision.models.resnet50(pretrained=True)
    model2 = torchvision.models.resnet50(pretrained=True)

    # model = FinalModel(model1, model2, 0.8).to(device)
    model = FinalModel(model1, model2).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    weight_decay = 0.3  # Regularization coefficient, which can be adjusted as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,  weight_decay=weight_decay)

    # Train the model
    model = train(model, train_loader, test_loader, val_loader, criterion, optimizer, device)
    save_predictions(model, train_loader, '/home/zhenyuan/mice/results/alldata_resnet50_merged_train_predictions.csv', device)
    # Save the validation set sample name and predicted probability
    save_predictions(model, val_loader, '/home/zhenyuan/mice/results/alldata_resnet50_merged_val_predictions.csv', device)
    # Save the test set sample name and predicted probability
    save_predictions(model, test_loader,'/home/zhenyuan/mice/results/alldata_resnet50_merged_test_predictions.csv', device)

if __name__ == "__main__":
    main()

