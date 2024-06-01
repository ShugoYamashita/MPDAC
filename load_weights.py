import torch
from models.ImageRestorationModel import ImageRestorationModel
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.utils as utils
load_path = '/Users/shugo/study/public_code/weights/MPDAC/renamed/Large_AllWeather.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:\t{}".format(device))

model = ImageRestorationModel(model_size='large')
model = ImageRestorationModel(model_size='small')


model = model.to(device)

model.eval()

checkpoint = torch.load(load_path)

# breakpoint()

model.load_state_dict(checkpoint)

input_image_path = '/Users/shugo/study/public_code/imgs/27_rain.png'
input_img = Image.open(input_image_path)
transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_gt = Compose([ToTensor()])
input_im = transform_input(input_img.unsqueeze(0))

restored_image = model(input_im)


utils.save_image(restored_image[0], '/Users/shugo/study/public_code/imgs/output.png')
