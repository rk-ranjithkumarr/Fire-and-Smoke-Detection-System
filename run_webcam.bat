@echo off
call venv\Scripts\activate
python fire_smoke_pro.py --source 0 --conf 0.35 --imgsz 1280
pause
