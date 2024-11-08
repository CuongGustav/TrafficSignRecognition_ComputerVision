from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Tải mô hình YOLO và CNN
yolo_model = YOLO('./Trainyolo/best.pt')
cnn_model = load_model('model_tt.h5')

# Dictionary để ánh xạ các lớp dự đoán với tên biển báo giao thông
classes = { 
    1:'Speed limit (20km/h)',
    2:'Speed limit (30km/h)', 
    3:'Speed limit (50km/h)', 
    4:'Speed limit (60km/h)', 
    5:'Speed limit (70km/h)', 
    6:'Speed limit (80km/h)', 
    7:'End of speed limit (80km/h)', 
    8:'Speed limit (100km/h)', 
    9:'Speed limit (120km/h)', 
    10:'No passing', 
    11:'No passing veh over 3.5 tons', 
    12:'Right-of-way at intersection', 
    13:'Priority road', 
    14:'Yield', 
    15:'Stop', 
    16:'No vehicles', 
    17:'Veh > 3.5 tons prohibited', 
    18:'No entry', 
    19:'General caution', 
    20:'Dangerous curve left', 
    21:'Dangerous curve right', 
    22:'Double curve', 
    23:'Bumpy road', 
    24:'Slippery road', 
    25:'Road narrows on the right', 
    26:'Road work', 
    27:'Traffic signals', 
    28:'Pedestrians', 
    29:'Children crossing', 
    30:'Bicycles crossing', 
    31:'Beware of ice/snow',
    32:'Wild animals crossing', 
    33:'End speed + passing limits', 
    34:'Turn right ahead', 
    35:'Turn left ahead', 
    36:'Ahead only', 
    37:'Go straight or right', 
    38:'Go straight or left', 
    39:'Keep right', 
    40:'Keep left', 
    41:'Roundabout mandatory', 
    42:'End of no passing', 
    43:'End no passing veh > 3.5 tons' 
}

# Tải font một lần
font_size = 80  # Điều chỉnh kích thước chữ tại đây
try:
    font = ImageFont.truetype("times new roman.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

# Hàm phân loại với CNN sau khi cắt từng vùng từ kết quả YOLO
def classify_region(image_region):
    image_region = image_region.resize((30, 30))
    image_array = np.expand_dims(np.array(image_region), axis=0)
    pred_probs = cnn_model.predict(image_array)
    pred_class = np.argmax(pred_probs, axis=1)
    return classes.get(pred_class[0] + 1, "Unknown")

# Hàm phát hiện và phân loại đối tượng trong hình ảnh
def detect_and_classify(image_path):
    # Phát hiện đối tượng bằng YOLO
    results = yolo_model(image_path)
    original_image = Image.open(image_path)
    image_draw = ImageDraw.Draw(original_image)

    # Duyệt qua các vùng đối tượng phát hiện được và phân loại ngay
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])  # Các tọa độ của bounding box
            image_region = original_image.crop((x1, y1, x2, y2))  # Cắt vùng đối tượng

            # Phân loại vùng đối tượng
            class_name = classify_region(image_region)

            # Tính toán kích thước của nhãn (sử dụng textbbox)
            text_bbox = image_draw.textbbox((x1, y1), class_name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Kiểm tra vị trí để nhãn không bị cắt
            label_y = y1 - text_height - 10
            if label_y < 0:
                label_y = y2 + 10

            # Vẽ khung và nhãn trên ảnh
            image_draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            image_draw.text((x1, label_y), class_name, fill="red", font=font)

    # Hiển thị ảnh kết quả
    original_image.show()

# Sử dụng hàm detect_and_classify với đường dẫn ảnh
image_path = "Trainyolo/ảnh/ảnh/z5862610237462_68525c1742a4311ad2341517d149468d.jpg"
detect_and_classify(image_path)
