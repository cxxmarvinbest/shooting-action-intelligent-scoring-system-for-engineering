import cv2

gst_pipeline = (
    "rtspsrc location=rtsp://admin:siboasi123@192.168.8.89:554/h264/ch1/main/av_stream "
    "latency=0 rtsp-transport=tcp ! "
    "rtph264depay ! queue max-size-buffers=1 leaky=downstream ! "
    "h264parse config-interval=-1 ! queue max-size-buffers=1 leaky=downstream ! "
    "mppvideodec ! queue max-size-buffers=1 leaky=downstream ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
