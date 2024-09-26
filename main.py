from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load YOLO models
yolo_sign_model = YOLO('./best.pt')  # Mô hình phát hiện biển báo
yolo_object_model = YOLO('./best2.pt')  # Mô hình phân loại đối tượng

# Hàm phát hiện và hiển thị
def detect_and_display(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Phát hiện biển báo giao thông
    results_signs = yolo_sign_model(image_path)

    # Vẽ bounding boxes cho biển báo giao thông
    for result in results_signs:
        for box in result.boxes:  # Truy cập vào boxes
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Chuyển đổi thành danh sách và lấy tọa độ

            # Crop ảnh trong bounding box
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

            # Phân loại đối tượng trong bounding box
            predictions = yolo_object_model(cropped_img)  # Dùng mô hình phân loại cho ảnh đã cắt
            
            # Biến để kiểm tra xem có nhãn nào hợp lệ không
            has_label = False
            
            # Lặp qua các dự đoán và vẽ nhãn
            for pred in predictions:
                for box in pred.boxes:
                    # Lấy điểm tin cậy và lớp dự đoán
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    # Kiểm tra nếu confidence đủ lớn
                    if confidence > 0.2:  # Chỉ vẽ nếu confidence > 0.2
                        # Lấy tên lớp từ chính mô hình (thuộc tính 'names')
                        label_name = yolo_object_model.names[class_id]
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)  # Vẽ box màu xanh
                        cv2.putText(img, f"{label_name}: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)  # Vẽ nhãn
                        has_label = True  # Đánh dấu là có nhãn hợp lệ

            # Nếu không có nhãn hợp lệ thì không vẽ bounding box cho biển báo giao thông

    # Hiển thị hình ảnh với bounding boxes và nhãn
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Phát hiện đối tượng và biển báo giao thông')
    plt.axis('off')  # Tắt trục để hiển thị sạch hơn
    plt.show()

# Đường dẫn tới hình ảnh
image_path = 't3.jpg'

# Phát hiện và hiển thị
detect_and_display(image_path)