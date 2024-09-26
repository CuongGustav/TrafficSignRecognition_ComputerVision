from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO('./best2.pt')

# Function to perform object detection and show the result
def detect_and_show(image_path):
    results = model(image_path)
    for result in results:
        # Chuyển kết quả về dạng ảnh
        in_arr = result.plot()
        image = Image.fromarray(in_arr[..., ::-1])
        
        # Hiển thị ảnh đã nhận diện đối tượng
        plt.imshow(image)
        plt.axis('off')  # Tắt hiển thị trục
        plt.show()

# Nhập tên ảnh và thực hiện nhận diện
image_file = 'crop/cropped_2.jpg'  # Thay thế 'image.jpg' bằng tên file của bạn
detect_and_show(image_file)
