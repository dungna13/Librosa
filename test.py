!pip install moviepy

from moviepy import VideoFileClip

# Đường dẫn đến file mp4
mp4_file_path = '/kaggle/input/videotrain12312/MB3R333) - Poppy Playtime 2.mp4'
# Đường dẫn để lưu file wav đầu ra
wav_file_path = '/kaggle/working/audio_output/Poppy2.wav'

# Tạo thư mục nếu chưa tồn tại
import os
output_folder = os.path.dirname(wav_file_path)
os.makedirs(output_folder, exist_ok=True)

# Chuyển đổi từ mp4 sang wav
video = VideoFileClip(mp4_file_path)
video.audio.write_audiofile(wav_file_path)

print(f"Converted {mp4_file_path} to {wav_file_path}")


def split_audio_to_segments(audio_path, segment_duration=5, npy_folder='audio_segments'):
    # Tạo thư mục để lưu các đoạn âm thanh nếu chưa tồn tại
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)

    # Tải âm thanh
    audio_data, sr = librosa.load(audio_path, sr=None)

    # Tính toán số mẫu cho mỗi đoạn
    segment_samples = segment_duration * sr

    # Tính toán số lượng đoạn âm thanh
    num_segments = len(audio_data) // segment_samples

    for segment_index in range(num_segments):
        start_sample = segment_index * segment_samples
        end_sample = start_sample + segment_samples
        segment_array = audio_data[start_sample:end_sample]

        # Lưu đoạn âm thanh dưới dạng numpy array
        np.save(os.path.join(npy_folder, f'segment_{segment_index}.npy'), segment_array)

    print(f"Đã lưu {num_segments} đoạn âm thanh thành công.")

# Đường dẫn tới file âm thanh
audio_path = '/kaggle/working/audio_output/Poppy2.wav'  # Thay đổi đường dẫn đến file .wav của bạn
npy_folder = '/kaggle/working/audio_segments2'

# Cắt âm thanh thành các đoạn 5 giây và lưu dưới dạng .npy
split_audio_to_segments(audio_path, segment_duration=5, npy_folder=npy_folder)


# Tải mô hình đã được huấn luyện
model = tf.keras.models.load_model('/kaggle/working/trained_model.h5')

# Thư mục chứa các file .npy
npy_folder = '/kaggle/working/audio_segments2'

# Hàm chuyển đổi âm thanh thành spectrogram
def audio_to_spectrogram(audio_data, sr):
    # Tạo spectrogram
    stft = np.abs(librosa.stft(audio_data))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    return spectrogram

# Danh sách để lưu các file chứa âm thanh abnormal và điểm predict_score
abnormal_files_with_scores = []

# Tạo hoặc mở file CSV để lưu kết quả
csv_file = '/kaggle/working/abnormal_predictions.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Ghi tiêu đề cột
    writer.writerow(["File", "Predict_Score"])

    # Duyệt qua từng file trong thư mục
    for file_name in os.listdir(npy_folder):
        if file_name.endswith('.npy'):
            # Tải file .npy
            segment_data = np.load(os.path.join(npy_folder, file_name))

            # Tính toán tần số lấy mẫu (sampling rate)
            sr = 22050  # Thay đổi nếu cần

            # Chuyển đổi âm thanh thành spectrogram
            spectrogram = audio_to_spectrogram(segment_data, sr)

            # Reshape cho phù hợp với đầu vào của mô hình
            # Thay đổi kích thước spectrogram để có kích thước 224x224
            spectrogram_resized = cv2.resize(spectrogram, (224, 224))
            
            # Nếu mô hình sử dụng 3 kênh màu, bạn có thể lặp lại spectrogram
            spectrogram_resized = np.stack((spectrogram_resized,) * 3, axis=-1)  # Thay đổi kích thước để có 3 kênh
            spectrogram_resized = np.expand_dims(spectrogram_resized, axis=0)  # Thêm chiều batch

            # Dự đoán nhãn cho đoạn âm thanh
            prediction = model.predict(spectrogram_resized)

            # Lấy xác suất cho nhãn "abnormal" (giả định nhãn abnormal là giá trị đầu tiên trong prediction)
            predict_score = prediction[0][0]

            # Xác định nhãn
            if predict_score > 0.2:  # Nếu mô hình trả về xác suất lớn hơn 0.5 cho nhãn abnormal
                abnormal_files_with_scores.append((file_name, predict_score))
                print(f"File {file_name} chứa âm thanh abnormal với điểm predict_score: {predict_score}")

            # Ghi kết quả vào file CSV
            writer.writerow([file_name, predict_score])

# In ra danh sách các file abnormal và điểm predict_score
print("Các file chứa âm thanh abnormal và điểm predict_score:", abnormal_files_with_scores)



# Đọc dữ liệu từ file CSV
csv_file = '/kaggle/working/abnormal_predictions.csv'
files = []
scores = []

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Bỏ qua hàng tiêu đề
    for row in reader:
        files.append(row[0])  # Tên file
        scores.append(float(row[1]))  # Điểm predict_score

# Hàm để trích xuất số từ tên file
def extract_segment_number(file_name):
    match = re.search(r'_(\d+)', file_name)
    return int(match.group(1)) if match else -1  # Trả về số đã trích xuất hoặc -1 nếu không tìm thấy

# Sắp xếp các file và scores theo số trong tên file
sorted_files_scores = sorted(zip(files, scores), key=lambda x: extract_segment_number(x[0]))
sorted_files, sorted_scores = zip(*sorted_files_scores)

# Vẽ đồ thị line graph
plt.figure(figsize=(10, 6))

# Vẽ đường biểu diễn scores
plt.plot(range(len(sorted_scores)), sorted_scores, marker='o', linestyle='-', color='b')

# Tô nền đỏ cho các điểm abnormal
abnormal_indices = [i for i, score in enumerate(sorted_scores) if score > 0.2]
for i in abnormal_indices:
    plt.axvspan(i - 0.4, i + 0.4, color='lightcoral', alpha=0.5)  # Tô màu nền

# Thêm tiêu đề và nhãn trục
plt.title('Predict Score for Audio Files', fontsize=16)
plt.xlabel('Audio Files', fontsize=14)
plt.ylabel('Predict Score', fontsize=14)

# Đánh dấu các điểm trên trục x với tên file đã sắp xếp
plt.xticks(range(len(sorted_files)), sorted_files, rotation=90, fontsize=10)

# Hiển thị đồ thị
plt.tight_layout()
plt.show()
