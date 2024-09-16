import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import setproctitle
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)

# Define initial joint positions for the cartoon character
character_joints = {
    'nose': (120, 100),
    'left_eye': (100, 90),
    'right_eye': (140, 90),
    'left_ear': (80, 90),
    'right_ear': (160, 90),
    'left_shoulder': (100, 200),
    'right_shoulder': (150, 200),
    'left_elbow': (80, 300),
    'right_elbow': (170, 300),
    'left_wrist': (60, 400),
    'right_wrist': (190, 400),
    'left_hip': (100, 400),
    'right_hip': (150, 400),
    'left_knee': (90, 500),
    'right_knee': (160, 500),
    'left_ankle': (80, 600),
    'right_ankle': (170, 600)
}

def draw_kinetic_character(joints):
    # Create a black background
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Draw head (circle) with eyes, nose, and mouth
    head_center_x = (joints['left_eye'][0] + joints['right_eye'][0]) // 2
    head_center_y = (joints['left_eye'][1] + joints['right_eye'][1]) // 2
    head_radius = 60
    cv2.circle(frame, (head_center_x, head_center_y), head_radius, (255, 255, 255), 2)  # Head outline

    # Draw eyes
    cv2.circle(frame, joints['left_eye'], 10, (255, 255, 255), -1)  # Left eye
    cv2.circle(frame, joints['right_eye'], 10, (255, 255, 255), -1)  # Right eye

    # Draw nose
    cv2.circle(frame, joints['nose'], 8, (255, 255, 255), -1)  # Nose

    # Draw mouth
    mouth_start = (head_center_x - 20, head_center_y + 30)
    mouth_end = (head_center_x + 20, head_center_y + 30)
    cv2.line(frame, mouth_start, mouth_end, (255, 255, 255), 2)  # Mouth

    # Draw body (lines connecting joints)
    cv2.line(frame, joints['left_shoulder'], joints['left_hip'], (255, 255, 255), 5)  # Left body line
    cv2.line(frame, joints['right_shoulder'], joints['right_hip'], (255, 255, 255), 5)  # Right body line
    cv2.line(frame, joints['left_shoulder'], joints['right_shoulder'], (255, 255, 255), 5)  # Shoulder line
    cv2.line(frame, joints['left_hip'], joints['right_hip'], (255, 255, 255), 5)  # Hip line

    # Draw arms (lines)
    cv2.line(frame, joints['left_shoulder'], joints['left_elbow'], (255, 255, 255), 5)  # Left arm
    cv2.line(frame, joints['left_elbow'], joints['left_wrist'], (255, 255, 255), 5)  # Left forearm
    cv2.line(frame, joints['right_shoulder'], joints['right_elbow'], (255, 255, 255), 5)  # Right arm
    cv2.line(frame, joints['right_elbow'], joints['right_wrist'], (255, 255, 255), 5)  # Right forearm

    # Draw legs (lines)
    cv2.line(frame, joints['left_hip'], joints['left_knee'], (255, 255, 255), 5)  # Left thigh
    cv2.line(frame, joints['left_knee'], joints['left_ankle'], (255, 255, 255), 5)  # Left leg
    cv2.line(frame, joints['right_hip'], joints['right_knee'], (255, 255, 255), 5)  # Right thigh
    cv2.line(frame, joints['right_knee'], joints['right_ankle'], (255, 255, 255), 5)  # Right leg

    # Draw joints as larger dots
    for key, point in joints.items():
        cv2.circle(frame, point, 12, (255, 255, 255), -1)  # Draw larger dots at joint positions
    
    return frame

# User-defined class to be used in the callback function
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

# User-defined callback function
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        label = detection.get_label()
        if label == "person":
            bbox = detection.get_bbox()  # Extracting the bounding box here
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()

                keypoint_indices = {
                    'nose': 0,
                    'left_eye': 1,
                    'right_eye': 2,
                    'left_ear': 3,
                    'right_ear': 4,
                    'left_shoulder': 5,
                    'right_shoulder': 6,
                    'left_elbow': 7,
                    'right_elbow': 8,
                    'left_wrist': 9,
                    'right_wrist': 10,
                    'left_hip': 11,
                    'right_hip': 12,
                    'left_knee': 13,
                    'right_knee': 14,
                    'left_ankle': 15,
                    'right_ankle': 16
                }

                for name, idx in keypoint_indices.items():
                    x = int((points[idx].x() * bbox.width() + bbox.xmin()) * 1280)
                    y = int((points[idx].y() * bbox.height() + bbox.ymin()) * 720)
                    character_joints[name] = (x, y)

    # Draw kinetic character on a black background
    frame = draw_kinetic_character(character_joints)
    user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

# User Gstreamer Application
class GStreamerPoseEstimationApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolov8pose_post.so')
        self.post_function_name = "filter"
        self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_pose_h8l_pi.hef')
        self.app_callback = app_callback
        setproctitle.setproctitle("Hailo Pose Estimation App")
        self.create_pipeline()

    def get_pipeline_string(self):
        if self.source_type == "rpi":
            source_element = f"libcamerasrc name=src_0 ! "
            source_element += f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
            source_element += QUEUE("queue_src_scale")
            source_element += f"videoscale ! "
            source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "
        elif self.source_type == "usb":
            source_element = f"v4l2src device={self.video_source} name=src_0 ! "
            source_element += f"video/x-raw, width=640, height=480, framerate=30/1 ! "
            source_element += QUEUE("queue_scale")
            source_element += f"videoscale n-threads=2 ! "
            source_element += QUEUE("queue_src_convert")
            source_element += f"videoconvert n-threads=3 name=src_convert qos=false ! "
            source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "

        pipeline_string = "hailomuxer name=hmux "
        pipeline_string += source_element
        pipeline_string += "tee name=t ! "
        pipeline_string += QUEUE("bypass_queue", max_size_buffers=20) + "hmux.sink_0 "
        pipeline_string += "t. ! " + QUEUE("queue_hailonet")
        pipeline_string += "videoconvert n-threads=3 ! "
        pipeline_string += f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} force-writable=true ! "
        pipeline_string += QUEUE("queue_hailofilter")
        pipeline_string += f"hailofilter function-name={self.post_function_name} so-path={self.default_postprocess_so} qos=false ! "
        pipeline_string += QUEUE("queue_hmuc") + " hmux.sink_1 "
        pipeline_string += "hmux. ! " + QUEUE("queue_hailo_python")
        pipeline_string += QUEUE("queue_user_callback")
        pipeline_string += f"identity name=identity_callback ! "
        pipeline_string += QUEUE("queue_hailooverlay")
        pipeline_string += f"hailooverlay ! "
        pipeline_string += QUEUE("queue_videoconvert")
        pipeline_string += f"videoconvert n-threads=3 qos=false ! "
        pipeline_string += QUEUE("queue_hailo_display")
        pipeline_string += f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    user_data = user_app_callback_class()
    parser = get_default_parser()
    args = parser.parse_args()
    app = GStreamerPoseEstimationApp(args, user_data)
    app.run()
