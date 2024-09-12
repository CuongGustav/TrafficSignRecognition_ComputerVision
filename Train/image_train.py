import os
import shutil

# Đường dẫn đến thư mục chứa các thư mục con
source_dir = 'D:/TrafficSignRecognition_ComputerVision/Train/after'

# Đường dẫn đến thư mục đích
destination_dir = 'D:/TrafficSignRecognition_ComputerVision/Train/coco128/images/train2017'

# Tạo thư mục đích nếu nó chưa tồn tại
os.makedirs(destination_dir, exist_ok=True)

# Khởi tạo số đếm cho các file ảnh
image_counter = 1

# Lặp qua tất cả các thư mục con trong thư mục nguồn
for subdir, _, files in os.walk(source_dir):
    for file in files:
        # Xây dựng đường dẫn đầy đủ cho file ảnh nguồn
        file_path = os.path.join(subdir, file)
        
        # Tạo tên mới cho file ảnh
        new_filename = f'image{image_counter}.jpg'
        
        # Xây dựng đường dẫn đầy đủ cho file ảnh đích
        destination_path = os.path.join(destination_dir, new_filename)
        
        # Sao chép file ảnh vào thư mục đích với tên mới
        shutil.copy(file_path, destination_path)
        
        # Tăng số đếm
        image_counter += 1

print('Sao chép và đổi tên các ảnh đã hoàn tất.')
