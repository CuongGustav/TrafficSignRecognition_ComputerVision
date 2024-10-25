from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO('./Trainyolo/best.pt')

# Function to perform object detection and display the result
def detect_objects(image_path):
    results = model(image_path)
    for result in results:
        in_arr = result.plot()
        image = Image.fromarray(in_arr[..., ::-1])
        # Hiển thị ảnh có đối tượng đã nhận diện
        image.show()

# Đường dẫn trực tiếp đến ảnh
image_path = './Trainyolo/ảnh/ảnh/z5862609985578_c78bf34fca1ae7c1841ffeda8728f1e6.jpg'
detect_objects(image_path)
