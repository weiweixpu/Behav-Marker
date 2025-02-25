from torchvision import transforms
from torch.utils.data import DataLoader
from models_mae import mae_vit_large_patch16
import torch
import torch.optim as optim
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm import tqdm
import datetime

def save_detailed_progress(info, file_path="/data/zhenyuan/pretrain/save_result/TM_detailed_training_progress.txt"):
    """
    Saves detailed progress information of each iteration to a specified file.

    Args:
        info (str): The progress information string to save.
        file_path (str): The path to the file where the progress will be saved.
    """
    with open(file_path, "a") as f:
        f.write(info + "\n")

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset by listing all image files in the specified directory.
        Args:
            data_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Ensure image is in RGB format.
        if self.transform:
            image = self.transform(image)
        return image

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

def save_epoch_info(epoch, timestamp):
    """
    Saves the epoch number and timestamp to a file
    """
    with open("/data/zhenyuan/pretrain/save_result/TM_epoch_info.txt", "a") as f:
        f.write(f"Epoch {epoch}: {timestamp}\n")

def load_previous_state(model, optimizer):
    """
    Loads the model and optimizer state from a checkpoint file if it exists.
    """
    if os.path.exists("/data/zhenyuan/pretrain/save_result/TM_model_checkpoint.pth"):
        checkpoint = torch.load("/data/zhenyuan/pretrain/save_result/TM_model_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
        return epoch, loss
    else:
        return 0, float('inf')

def save_encoder_parameters(model, save_path):
    # Get all encoder parameters
    encoder_state_dict = model.state_dict()

    # Filter based on the prefix of the encoder parameter name
    # Assume that all encoder parameters are prefixed with "encoder."
    encoder_keys = [k for k in encoder_state_dict.keys() if not k.startswith('decoder')]

    # Exclude decoder parameters and keep only encoder parameters
    encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if k in encoder_keys}

    # Save the encoder state dictionary
    torch.save(encoder_state_dict, save_path)

def save_checkpoint(epoch, model, optimizer, loss, last_epoch):
    """
    Saves the current state of the model, optimizer, and the last epoch's loss.
    Deletes the checkpoint of the last epoch.
    """
    last_encoder_checkpoint_filename = f"/data/zhenyuan/pretrain/save_result/TM_encoder_checkpoint_{last_epoch}.pth"
    last_checkpoint_filename = f"/data/zhenyuan/pretrain/save_result/TM_model_checkpoint_{last_epoch}.pth"
    if os.path.exists(last_checkpoint_filename):
        os.remove(last_checkpoint_filename)

    if os.path.exists(last_encoder_checkpoint_filename):
        os.remove(last_encoder_checkpoint_filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, f"/data/zhenyuan/pretrain/save_result/TM_model_checkpoint_{epoch}.pth")


def adjust_learning_rate(optimizer, epoch, warmup_epochs=3, initial_lr=0.001):
    """Adjusts the learning rate according to the epoch number."""
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        lr = initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


dataset = CustomImageDataset("/data/zhenyuan/pretrain/trajectory/", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = mae_vit_large_patch16().cuda(0)

num_epochs = 100
warmup_epochs = 3
initial_lr = 0.001
last_epoch = -1


optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

start_epoch, best_loss = load_previous_state(model, optimizer)

for epoch in range(start_epoch, num_epochs):
    adjust_learning_rate(optimizer, epoch, warmup_epochs, initial_lr)  # Dynamically adjust the learning rate
    start_time = datetime.datetime.now()
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
    for imgs in progress_bar:
        imgs = imgs.cuda(0)
        optimizer.zero_grad()
        loss, _, _ = model(imgs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / len(dataloader))

    scheduler.step()  # Decay learning rate
    end_time = datetime.datetime.now()
    save_epoch_info(epoch + 1, end_time)
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    if (epoch + 1) % 1 == 0:
        save_checkpoint(epoch + 1, model, optimizer, total_loss / len(dataloader), last_epoch)
        last_epoch = epoch + 1
        save_encoder_parameters(model, f'/data/zhenyuan/pretrain/save_result/TM_encoder_checkpoint_{epoch}.pth')

model_save_path = '/data/zhenyuan/pretrain/save_result/vit_pretrain_TM.pth'
torch.save(model.state_dict(), model_save_path)
