import torch
import torchvision
from torchvision import transforms
from PIL import Image

# Category to be predicted
classes = ['Control', 'PD']

def predict_class(img_path, model):
    img = Image.open(img_path)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).cuda()
    img = torch.unsqueeze(img, dim=0)
    out = model(img)
    # print('out = ', out)
    pre = torch.max(out, 1)[1]
    cls = classes[pre.item()]
    print('It is {}!'.format(cls))

def model_struct(num_cls):
    model_densenet = torchvision.models.densenet121(pretrained=True)
    num_fc = model_densenet.classifier.in_features
    model_densenet.classifier = torch.nn.Linear(num_fc, num_cls)
    for param in model_densenet.parameters():
        param.requires_grad = False
    for param in model_densenet.classifier.parameters():
        param.requires_grad = True
    model_densenet.to('cuda')
    return model_densenet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_struct(2)
    model.to(device)
    model.eval()
    save = torch.load('/home/lzr/hzy/DenseNet121/model.pth')  # The weight you want to call
    model.load_state_dict(save['model'])
    img = 'PD.png'  # The image to be detected
    predict_class(img, model)

if __name__ == '__main__':
    main()