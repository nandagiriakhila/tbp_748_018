import cv2
import csv
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Function to perform prediction
def predict(chosen_model, img, classes=[], conf=0.3):  # Lower confidence threshold
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

# Function to perform detection and draw bounding boxes
def predict_and_detect(chosen_model, img, classes=[], conf=0.3):  # Lower confidence threshold
    results = predict(chosen_model, img, classes, conf)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            print(f"Detected object: {class_name} with confidence: {box.conf[0]}")  # Debug information
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{class_name} {box.conf[0]:.2f}",  # Add confidence score to the label
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

# Function to log detection results to CSV
def log_results_to_csv(results, output_csv):
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header row
            writer.writerow(["Object", "X1", "Y1", "X2", "Y2", "Confidence"])
        for result in results:
            for box in result.boxes:
                writer.writerow([result.names[int(box.cls[0])], 
                                 int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                                 int(box.xyxy[0][2]), int(box.xyxy[0][3]),
                                 float(box.conf[0])])

# Function to create a video writer
def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    return writer

# Function to process video
def process_video(video_path, output_csv, output_filename):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    writer = create_video_writer(cap, output_filename)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        result_img, results = predict_and_detect(model, img, classes=[], conf=0.3)  # Lower confidence threshold
        log_results_to_csv(results, output_csv)
        
        writer.write(result_img)
        cv2.imshow("Image", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

# Function to process image
def process_image(image_path, output_csv, output_filename):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open image {image_path}")
        return

    result_img, results = predict_and_detect(model, img, classes=[], conf=0.3)  # Lower confidence threshold
    log_results_to_csv(results, output_csv)
    
    cv2.imshow("Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the image outside of cv2.imshow to avoid filename issues
    cv2.imwrite(output_filename, result_img)

# Main function to handle input type
def main(input_path, output_csv, output_filename):
    if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(input_path, output_csv, output_filename)
    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png','.webp')):
        process_image(input_path, output_csv, output_filename)
    else:
        print(f"Unsupported file type: {input_path}")

# Example usage
input_path = r"C:\hyyyy\realtimeimage.jpeg"  # Change to your input file path
#output_csv = r"C:\hyyyy\detected_objects.csv"
output_csv = r"C:\hyyyy\new.csv"
output_filename = r"C:\hyyyy\output.jpg"  # Change to .mp4 if processing video

main(input_path, output_csv, output_filename)
