import cv2
import pandas as pd
from ultralytics import YOLO
import math
import torch
import os
import numpy as np
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import threading
import time
from collections import defaultdict
import warnings
from flask import send_file  


import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential


torch.serialization.add_safe_globals([DetectionModel, Sequential])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['SECRET_KEY'] = 'your-secret-key-here'


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


class_mapping = {
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Big Bus",
    "truck": "2-Axle Truck",
}


vehicle_categories = [
    "Car",
    "Motorcycle",
    "Auto (Three-Wheeler)",
    "Small Bus",
    "Big Bus",
    "Light Commercial Vehicle (LCV)",
    "2-Axle Truck",
    "3-Axle Truck",
    "Multi-Axle Truck",
    "Tractor-Trailer",
    "Multi-Axle Vehicles (More Than 3 Axles)",
    "Ambulance"
]


class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.vehicle_counts = {category: 0 for category in vehicle_categories}
        self.vehicle_tracks = {}
        self.counting_line_pos = 184
        self.offset = 8
        self.crossed_ids = set()

    def update(self, objects_rect, class_names):
        objects_bbs_ids = []
        
        for rect, class_name in zip(objects_rect, class_names):
            x1, y1, x2, y2 = rect
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)
            
           
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if class_name == "bus":
                if area < 5000:
                    class_name = "Small Bus"
                else:
                    class_name = "Big Bus"
            elif class_name == "truck":
                if area < 4000:
                    class_name = "Light Commercial Vehicle (LCV)"
                elif area < 8000:
                    class_name = "2-Axle Truck"
                else:
                    class_name = "Multi-Axle Truck"
            
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, id, class_name])
                    same_object_detected = True
                    break

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count, class_name])
                self.vehicle_tracks[self.id_count] = class_name
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    def count_vehicle(self, class_name, obj_id):
        if class_name in self.vehicle_counts and obj_id not in self.crossed_ids:
            self.vehicle_counts[class_name] += 1
            self.crossed_ids.add(obj_id)

    def get_counts(self):
        return self.vehicle_counts

    def get_tracks(self):
        return self.vehicle_tracks


class_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def download_yolo_weights():
    if not os.path.exists('yolov8s.pt'):
        import urllib.request
        url = 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt'
        urllib.request.urlretrieve(url, 'yolov8s.pt')

download_yolo_weights()


processing_status = {
    'is_processing': False,
    'filename': None,
    'progress': 0,
    'results': None
}

def safe_load_model():
    try:
       
        with torch.serialization.safe_globals([DetectionModel, Sequential]):
            model = YOLO('yolov8s.pt')
            return model
    except Exception as e:
        warnings.warn(f"Safe load failed: {str(e)}")
        try:
           
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
            model = YOLO('yolov8s.pt')
            torch.load = original_load  
            return model
        except Exception as e:
            warnings.warn(f"Fallback load failed: {str(e)}")
            raise

def process_video(video_path, output_path):
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['filename'] = os.path.basename(video_path)
        processing_status['progress'] = 0
        
        model = safe_load_model()
        
        tracker = Tracker()
        cap = cv2.VideoCapture(video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1020, 500))
        
        frame_count = 0
        processing_start = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 3 != 0:
                continue
            
            processing_status['progress'] = min(99, int((frame_count / total_frames) * 100))
            
            frame = cv2.resize(frame, (1020, 500))
            results = model.predict(frame, verbose=False)
            
            vehicles = []
            vehicle_classes = []
            
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    class_id = box.cls[0].astype(int)
                    conf = box.conf[0]
                    
                    if conf < 0.5:
                        continue
                        
                    class_name = class_list[class_id]
                    
                    if class_name in ["car", "bus", "truck", "motorcycle"]:
                        mapped_class = class_mapping.get(class_name, class_name)
                        vehicles.append([x1, y1, x2, y2])
                        vehicle_classes.append(mapped_class)

            tracked_objects = tracker.update(vehicles, vehicle_classes)
            cv2.line(frame, (1, tracker.counting_line_pos), (1018, tracker.counting_line_pos), (0, 255, 0), 2)

            for bbox in tracked_objects:
                x1, y1, x2, y2, obj_id, class_name = bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                if (cy > tracker.counting_line_pos - tracker.offset) and (cy < tracker.counting_line_pos + tracker.offset):
                    tracker.count_vehicle(class_name, obj_id)

                if "Car" in class_name:
                    color = (0, 255, 0)
                elif "Bus" in class_name:
                    color = (0, 0, 255)
                elif "Truck" in class_name:
                    color = (255, 0, 0)
                elif "Motorcycle" in class_name:
                    color = (255, 255, 0)
                else:
                    color = (255, 0, 255)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ID:{obj_id}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset = 40
            for i, (category, count) in enumerate(tracker.get_counts().items()):
                if count > 0:
                    cv2.putText(frame, f'{category}: {count}', (20, y_offset + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(frame)

        processing_end = datetime.now()
        processing_time = (processing_end - processing_start).total_seconds()
        
        cap.release()
        out.release()
        
        final_counts = tracker.get_counts()
        total_vehicles = sum(final_counts.values())
        
        counts_path = os.path.join(app.config['OUTPUT_FOLDER'], 'vehicle_counts.json')
        with open(counts_path, 'w') as f:
            json.dump(final_counts, f, indent=4)
        
        report_content = f"""
        Vehicle Detection and Classification Report
        -----------------------------------------

        1. Model Information:
        - Model: YOLOv8s (pretrained on COCO dataset)
        - Custom classification: Added size-based classification for bus and truck types

        2. Processing Metrics:
        - Video duration: {total_frames/fps:.2f} seconds
        - Total frames processed: {frame_count}
        - Processing time: {processing_time:.2f} seconds
        - Processing speed: {frame_count/processing_time:.2f} FPS

        3. Vehicle Counts:
        {json.dumps(final_counts, indent=4)}

        4. Limitations:
        - Limited to COCO classes (missing some specific vehicle types)
        - Size-based classification may not be perfect
        """
        
        report_path = os.path.join(app.config['OUTPUT_FOLDER'], 'report.txt')
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        processing_status['results'] = {
            'video_path': output_path,
            'counts_path': counts_path,
            'report_path': report_path,
            'processing_time': processing_time,
            'total_vehicles': total_vehicles,
            'vehicle_counts': final_counts
        }
        processing_status['progress'] = 100
        processing_status['is_processing'] = False
        
        return processing_status['results']
    
    except Exception as e:
        print(f"Error processing video: {e}")
        processing_status['is_processing'] = False
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_progress')
def get_progress():
    return jsonify({
        'is_processing': processing_status['is_processing'],
        'progress': processing_status['progress'],
        'filename': processing_status['filename'],
        'has_results': processing_status['results'] is not None,
        'results': processing_status['results']
    })

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        output_filename = f"annotated_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        global processing_status
        processing_status = {
            'is_processing': True,
            'filename': filename,
            'progress': 0,
            'results': None
        }
        
        thread = threading.Thread(target=process_video, args=(upload_path, output_path))
        thread.start()
        
        return jsonify({'success': True, 'message': 'Video processing started'})
    
    return jsonify({'success': False, 'message': 'Invalid file format'})

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/download_report')
def download_report():
    if not processing_status['results']:
        return "Report not available", 404
    report_path = os.path.join(app.config['OUTPUT_FOLDER'], 'report.txt')
    return send_file(report_path, as_attachment=True)

@app.route('/download_video')
def download_video():
    if not processing_status['results']:
        return "Video not available", 404
    video_path = processing_status['results']['video_path']
    return send_file(video_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)