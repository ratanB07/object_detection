# object_detection
🚦 Vehicle Detection and Classification System
Real-time vehicle tracking and classification using YOLOv8 + Flask + OpenCV


📌 Overview
This project is a video-based traffic monitoring system that detects, classifies, and counts various types of vehicles using YOLOv8. The web interface is built using Flask, and it supports real-time video processing and result visualization.

📽️ Key Features
⚡ Fast vehicle detection with YOLOv8

📊 Intelligent classification based on size & shape

🧠 Custom logic for distinguishing trucks, buses, LCVs

📁 Upload and process video files via web UI

📉 Auto-generated PDF reports & JSON data

🧾 Download annotated video & detailed statistics

🚗 Supported Vehicle Types
Category	Notes
Car, Motorcycle	COCO-pretrained detection
Bus	Separated into Small Bus & Big Bus based on area
Truck	Split into LCV, 2-Axle, Multi-Axle, etc.
Ambulance, Auto Rickshaw	Custom classification added

🛠️ Tech Stack
Component	Purpose
YOLOv8	Real-time object detection
OpenCV	Frame extraction, drawing, and annotation
Flask	Web-based video uploader and API routes
ReportLab	Generate professional PDF reports
Multithreading	Non-blocking video processing

💡 Advantages
✅ No need to retrain the model — uses pretrained COCO classes

✅ Handles real-time and offline video processing

✅ Dynamic vehicle classification logic improves accuracy

✅ Reports stored in output/ as JSON and downloadable PDFs

✅ Clean web UI and scalable backend for future enhancements


🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/ratanB07/object_detection.git
cd object_detection
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Download YOLOv8 Weights (Auto-downloaded on first run)
Or manually place yolov8s.pt in the project root.

4. Run the Flask Server
bash
Copy
Edit
python app.py
5. Open in Browser
Go to http://127.0.0.1:5000 to use the app.

📊 Output Structure
bash
Copy
Edit
├── uploads/               # Uploaded videos
├── output/
│   ├── annotated_video.mp4
│   ├── vehicle_counts.json
│   └── report.pdf
📥 Sample Report Preview

📌 Future Improvements
📍 Real-time camera stream support

📍 Dashboard analytics for vehicle flow

📍 Live map-based visualization

📍 Database integration (PostgreSQL / MongoDB)

📄 License
This project is licensed under the MIT License. Feel free to use and modify!

🙋‍♂️ Author
Your Name
[LinkedIn | Portfolio |](https://www.linkedin.com/in/ratan-biswakarmakar-7ab97317a/?trk=opento_sprofile_details)
ratanbisong@gmail.com
