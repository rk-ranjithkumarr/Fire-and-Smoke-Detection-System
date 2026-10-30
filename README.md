# Professional Fire & Smoke CCTV Detection

Features: YOLO fire/smoke detection, RTSP/webcam/video input, high-resolution inference,
persistent PC alarm, visual alert, alert snapshots, optional recording, CPU/GPU selection.

IMPORTANT: Software cannot guarantee detection at 200 m. Actual range depends on camera
resolution, optical zoom/lens, target size, lighting, haze, mounting and training data.
For a real 200 m deployment use a suitable high-resolution varifocal/telephoto IP camera,
validate on-site, and use certified fire/smoke alarm equipment as the primary life-safety
system.

INSTALL (Windows CMD)
1. py -3.11 -m venv venv
2. venv\Scripts\activate
3. python -m pip install --upgrade pip
4. pip install -r requirements.txt
5. python -c "import cv2; print(cv2.__version__); cv2.namedWindow('T'); cv2.destroyAllWindows(); print('OpenCV GUI OK')"

RUN
Webcam: python fire_smoke_pro.py --source 0
CCTV:   python fire_smoke_pro.py --source "rtsp://username:password@CAMERA_IP:554/stream"
Video:  python fire_smoke_pro.py --source "C:\videos\test.mp4"
GPU:    python fire_smoke_pro.py --source 0 --device 0
Record: python fire_smoke_pro.py --source 0 --save-video

The first run automatically downloads weights/best.pt.
Default sensitivity: confidence 0.35, image size 1280.
Lower confidence = more sensitivity but potentially more false alarms.
