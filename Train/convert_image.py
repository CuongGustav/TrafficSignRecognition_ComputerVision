from PIL import Image
import os

def process_images_in_folder(folder_path, output_base_path, new_width=640, new_format="jpg"):
    # Duyệt qua từng thư mục con trong thư mục chính
    for subfolder in os.listdir(folder_path):
        input_folder = os.path.join(folder_path, subfolder)

        # Kiểm tra xem đây có phải là thư mục không
        if os.path.isdir(input_folder):
            # Lấy phần tên sau dấu gạch dưới của thư mục con
            folder_name = os.path.basename(input_folder)
            base_name = folder_name.split('_')[-1].lower()
            
            # Tạo thư mục đầu ra tương ứng trong thư mục output
            output_folder = os.path.join(output_base_path, subfolder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Lấy danh sách các file ảnh trong thư mục
            file_list = os.listdir(input_folder)
            
            # Lặp qua từng file ảnh trong thư mục
            for index, filename in enumerate(file_list):
                file_path = os.path.join(input_folder, filename)

                # Mở file ảnh
                try:
                    img = Image.open(file_path)
                except Exception as e:
                    print(f"Không thể mở file {filename}: {e}")
                    continue

                # Lấy kích thước gốc của ảnh
                original_width, original_height = img.size

                # Tính toán chiều cao mới dựa trên tỉ lệ khung hình
                new_height = int((new_width / original_width) * original_height)

                # Đổi kích thước ảnh
                img = img.resize((new_width, new_height))

                # Tạo tên mới cho file ảnh (đuôi là .jpg)
                new_file_name = f"{base_name}{index + 1}.jpg"
                new_file_path = os.path.join(output_folder, new_file_name)

                # Lưu ảnh với định dạng JPEG
                img.convert('RGB').save(new_file_path, "JPEG")

                print(f"Processed {filename} -> {new_file_name} (new size: {new_width}x{new_height})")


# Đường dẫn thư mục chứa tất cả các thư mục con
input_folder_path = "D:/TrafficSignRecognition_ComputerVision/Train/before"
output_folder_path = "D:/TrafficSignRecognition_ComputerVision/Train/after"

# Thực hiện xử lý tất cả các thư mục con
process_images_in_folder(input_folder_path, output_folder_path)
