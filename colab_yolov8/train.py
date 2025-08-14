from ultralytics  import YOLO
import cv2

# Eğitilen modeli yükle
model = YOLO("model_colab.pt")  # Model dosyanızın yolunu buraya yazın

#model = YOLO(r"C:\Users\Tuğçe Karagöz\Desktop\colab_yolov8\model_colab.pt")

#model = YOLO(r"C:/Users/Tuğçe Karagöz/Desktop/colab_yolov8/model_colab.pt")


# Video dosyasını yükle

video_path ="ekmek_video.mp4"  # Video dosyanızın adını buraya yazın


#video_path = r"C:/Users/Tuğçe Karagöz/Desktop/colab_yolov8/video.mp4" 
cap = cv2.VideoCapture(video_path)

# Video kaydedici ayarları
output_path = "output_video.mp4"  # Çıktı videosunun adını belirleyin
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin yap
    results = model(frame)

    # Her bir sonuç için bounding box çizin
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Bounding box koordinatları
            conf = 0.7
            #conf = box.conf[0].item()  # Güven skoru
            cls = box.cls[0].item()  # Sınıf ID'si
            label = f"{model.names[int(cls)]} {conf:.2f}"  # Etiket ve güven skoru

            # Bounding box'u çizin
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sonuçları kaydedin
    out.write(frame)

    # İşlenmiş videoyu görüntüleyin (isteğe bağlı)
    cv2.imshow("YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırakın
cap.release()
out.release()
cv2.destroyAllWindows()
