import argparse,time,urllib.request
from datetime import datetime
from pathlib import Path
import cv2,numpy as np
from ultralytics import YOLO

MODEL_URL="https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt?download=true"
MODEL_PATH=Path("weights/best.pt"); ALERT_DIR=Path("logs/alerts")
FIRE={"fire","flame","flames"}; SMOKE={"smoke","smoke_cloud","smokecloud"}

def ensure_model():
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size>100000:return
    if MODEL_PATH.exists() and not MODEL_PATH.is_file():
        raise RuntimeError("weights\\best.pt is a folder. Delete it.")
    print("[INFO] Downloading Fire/Smoke YOLO model...")
    urllib.request.urlretrieve(MODEL_URL,str(MODEL_PATH))
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size<100000:
        MODEL_PATH.unlink(missing_ok=True); raise RuntimeError("Model download failed.")

def start_alarm():
    try:
        import pygame
        pygame.mixer.init(); pygame.mixer.music.load("alarm.wav")
        pygame.mixer.music.set_volume(1.0); pygame.mixer.music.play(-1); return pygame
    except Exception as e: print("[WARNING] Alarm unavailable:",e); return None

def stop_alarm(p):
    if p:
        try:p.mixer.music.stop();p.mixer.quit()
        except Exception:pass

def cname(names,i):
    return str(names.get(i,i) if isinstance(names,dict) else names[i]).lower().strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="0"); ap.add_argument("--conf",type=float,default=.35)
    ap.add_argument("--imgsz",type=int,default=1280); ap.add_argument("--device",default="cpu")
    ap.add_argument("--save-video",action="store_true"); a=ap.parse_args()
    ensure_model(); model=YOLO(str(MODEL_PATH))
    print("[INFO] Classes:",model.names,"| conf:",a.conf,"| imgsz:",a.imgsz,"| device:",a.device)
    src=int(a.source) if a.source.isdigit() else a.source
    cap=cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError("Cannot open source: "+a.source)
    ALERT_DIR.mkdir(parents=True,exist_ok=True); writer=None; alarm=None; last=0; n=0; t0=time.time()
    print("[INFO] Detection running. Press Q to quit.")
    try:
        while True:
            ok,frame=cap.read()
            if not ok: print("[INFO] Stream ended/disconnected."); break
            n+=1; r=model.predict(frame,conf=a.conf,imgsz=a.imgsz,device=a.device,verbose=False)[0]; found=[]
            if r.boxes is not None:
                for b in r.boxes:
                    i=int(b.cls[0].item()); c=float(b.conf[0].item()); name=cname(model.names,i)
                    danger=name in FIRE or name in SMOKE
                    if danger:found.append((name,c))
                    x1,y1,x2,y2=map(int,b.xyxy[0].tolist()); col=(0,0,255) if danger else (0,180,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),col,3)
                    cv2.putText(frame,f"{name.upper()} {c:.0%}",(x1,max(30,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,.8,col,2)
            if found:
                if alarm is None: alarm=start_alarm()
                cv2.rectangle(frame,(0,0),(frame.shape[1],78),(0,0,180),-1)
                cv2.putText(frame,"!!! FIRE / SMOKE ALERT !!!",(18,52),cv2.FONT_HERSHEY_SIMPLEX,1.15,(255,255,255),3)
                now=time.time()
                if now-last>=5:
                    last=now; p=ALERT_DIR/f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(p),frame); print("[ALERT]",[(x,round(y,3)) for x,y in found],"snapshot:",p)
            else:
                stop_alarm(alarm); alarm=None
                cv2.rectangle(frame,(0,0),(frame.shape[1],55),(20,120,20),-1)
                cv2.putText(frame,"STATUS: NORMAL",(18,39),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
            fps=n/max(time.time()-t0,1e-6)
            cv2.putText(frame,f"FPS {fps:.1f} | CONF {a.conf:.2f} | IMG {a.imgsz}",(10,frame.shape[0]-18),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            if a.save_video and writer is None:
                h,w=frame.shape[:2]; fi=cap.get(cv2.CAP_PROP_FPS)
                if not fi or np.isnan(fi) or fi<=0:fi=25
                writer=cv2.VideoWriter("logs/fire_smoke_detection_output.mp4",cv2.VideoWriter_fourcc(*"mp4v"),fi,(w,h))
            if writer:writer.write(frame)
            cv2.imshow("Professional Fire & Smoke CCTV Detection",frame)
            if cv2.waitKey(1)&0xFF in (ord("q"),ord("Q")):break
    finally:
        stop_alarm(alarm);cap.release()
        if writer:writer.release()
        cv2.destroyAllWindows()
if __name__=="__main__":main()
