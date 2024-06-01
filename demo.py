import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as utils
from PIL import Image
import os
import argparse
from models.ImageRestorationModel import ImageRestorationModel

def get_args_parser():
    parser = argparse.ArgumentParser(description='Demonstration of adverse weather removal')
    parser.add_argument('--input_image_path', type=str, help='Path to a input image', default='./imgs/degraded_imgs/sample_raindrop.png')
    parser.add_argument('--save_dir', type=str, help='Path to a save directory', default='./imgs/restored_imgs')
    parser.add_argument('--weights_path', type=str, help='Path to model weights', default='./weights/Small_AllWeather.pth')
    parser.add_argument('--model_size', type=str, help='model size', choices=['small', 'large'], default='small')
    return parser

def preprocess_image(image_path):
    """
    Preprocess the image by transforming to a tensor,
    normalizing, and adding a batch dimension.
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

def pad_img(img, img_multiple_of=32):
    """
    Reflect padding to ensure the image dimensions are multiples of 32.
    """
    height, width = img.shape[2], img.shape[3]

    H = (height + img_multiple_of - 1) // img_multiple_of * img_multiple_of
    W = (width + img_multiple_of - 1) // img_multiple_of * img_multiple_of
    padh = H - height
    padw = W - width

    img = F.pad(img, (0,padw,0,padh), 'reflect')

    return img, height, width

def main():
    args = get_args_parser()
    args = args.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:\t{}".format(device))

    model = ImageRestorationModel(model_size=args.model_size)
    model = model.to(device)
    model.eval()

    model.load_state_dict(torch.load(args.weights_path))
    input_image = preprocess_image(args.input_image_path)
    input_image, height, width = pad_img(input_image)
    input_image = input_image.to(device)

    with torch.no_grad():
        restored_image = model(input_image)

    restored_image = torch.clamp(restored_image, 0, 1)
    restored_image = restored_image[:, :, :height, :width]

    utils.save_image(restored_image[0], os.path.join(args.save_dir, os.path.basename(args.input_image_path)))

if __name__ == '__main__':
    main()
