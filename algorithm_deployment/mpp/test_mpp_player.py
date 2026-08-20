import sys
import cv2
import threading
import mpp_player
import time

# 全局变量保存最新帧
current_frame = None
frame_lock = threading.Lock()
is_rgb=False
rtsp="rtsp://admin:siboasi123@192.168.8.89:554/h264/ch1/main/av_stream"
display_width=1280
display_height=720
def frame_callback(frame_image,scale_image, frame_id,is_rgb):
    global current_frame
    with frame_lock:
        # 复制一份避免被覆写
        current_frame = frame_image.copy()
    # 也可在此直接显示，但最好在主线程中显示
    # print(f"Frame {frame_id}, shape={frame_np.shape}")
    if frame_id<=1:
        w = player.get_width()
        h = player.get_height()
        print(f"当前视频分辨率: {w}x{h}")

def error_callback(error):
    print(f"播放错误回调: {error}")

player = mpp_player.MppPlayer()
player.set_callback_frame(frame_callback)
player.set_callback_error(error_callback)
player.set_print_fps(True,100)
def play_thread():
    # 不显示窗口，只回调
    state=player.play(rtsp,display_width=display_width, display_height=display_height,is_rgb=is_rgb)
    if not state:
        print("播放结束")

# t = threading.Thread(target=play_thread, daemon=True)
# t.start()
play_thread()


# 主线程循环显示
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
while player.is_running():
    with frame_lock:
        if current_frame is not None:
            cv2.imshow("Video", current_frame)
            current_frame = None  # 消费掉
    if cv2.waitKey(1) == 27:  # ESC
        break
    time.sleep(0.01)
cv2.destroyAllWindows()
print('**1')
player.stop()
print('**2')
player.close()
# print('**3')
# player.close()
print("程序结束")