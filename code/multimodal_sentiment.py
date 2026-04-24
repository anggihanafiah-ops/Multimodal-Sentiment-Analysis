!pip install -q transformers accelerate bitsandbytes opencv-python pillow
pip install bitsandbytes accelerate qwen-vl-utils

## Connect google drive
from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive/Multimodal/"
VIDEO_PATH = BASE_PATH + "dataset_video/"
LABEL_PATH = BASE_PATH + "labels/"
