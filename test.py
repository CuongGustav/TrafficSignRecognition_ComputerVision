import os
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO('./best4.pt')

# Function to perform object detection on an image
def detect_objects(image_path):
    results = model(image_path)
    for result in results:
        in_arr = result.plot()
        image = Image.fromarray(in_arr[..., ::-1])
        # Lưu ảnh có đối tượng đã nhận dạng vào thư mục output
        output_path = os.path.join('output', os.path.basename(image_path))
        image.save(output_path)

# Thư mục chứa ảnh đầu vào
input_folder = 'image/test'

# Tạo thư mục output nếu chưa tồn tại
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua tất cả các tệp tin ảnh trong thư mục input_folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        image_path = os.path.join(input_folder, filename)
        detect_objects(image_path)
