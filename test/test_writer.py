import cv2
import numpy as np

print("==== OpenCV build info FFMPEG ====")
print(cv2.getBuildInformation())

W, H = 1280,720
fourcc_avc1 = cv2.VideoWriter_fourcc(*'avc1')
print(f"fourcc avc1 int={fourcc_avc1}")

out = cv2.VideoWriter("test_videos/left_rtsp.mp4", fourcc_avc1, 20.0, (W,H))
if not out.isOpened():
    print("❌ avc1 打开失败，降级测试 mp4v")
    fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("test_videos/left_rtsp.mp4", fourcc_mp4v,20.0,(W,H))

if out.isOpened():
    for _ in range(50):
        frm = np.random.randint(0,255,(H,W,3),dtype=np.uint8)
        out.write(frm)
    out.release()
    print("✅测试文件生成完成")
else:
    print("❌全部都打开失败")
