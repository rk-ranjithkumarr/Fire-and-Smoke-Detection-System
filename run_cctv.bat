@echo off
call venv\Scripts\activate
set /p RTSP=Enter CCTV RTSP URL: 
python fire_smoke_pro.py --source "%RTSP%" --conf 0.35 --imgsz 1280
pause
