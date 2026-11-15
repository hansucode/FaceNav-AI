import cv2
import os
import csv
import time
import math
import numpy as np
import mediapipe as mp
import urllib.request
import pyautogui
import platform

# SETTINGS
camera_number = 0
target_fps = 10
average_frames = 5
chin_extension = 50
mirror_camera = True
mouse_step = 40
output_file = "mediapipe_live_output.csv"
window_name = "FaceNav AI - Live"

# MOUSE SAFETY
mouse_enabled = False
window_enabled = True
profile_name = "Child"
mouth_toggle_ready = False
eyes_toggle_ready = False
wink_ready = False
last_mouse_action = "None"

# PROFILE THRESHOLDS
profiles = { "Adult": {"nose_min": 28, "nose_max": 44, "mouth_x_min": 1, "mouth_x_max": 10, "left_eye_open": 10, 
                       "right_eye_open": 10, "mouth_open": 25},
             "Child": {"nose_min": 18, "nose_max": 23, "mouth_x_min": -2, "mouth_x_max": 2, "left_eye_open": 5, 
                       "right_eye_open": 5, "mouth_open": 25} }

# CLICKABLE PROFILE SWITCH IN THE DISPLAY WINDOW
# Coordinates refer to the native 690 x 310 image, not the resized window.
profile_button = (435, 270, 555, 299)

def on_display_click(event, x, y, flags, userdata):
  global profile_name, wink_ready
  if event == cv2.EVENT_LBUTTONDOWN:
    x1, y1, x2, y2 = profile_button
    if x1 <= x <= x2 and y1 <= y <= y2:
      profile_name = "Child" if profile_name == "Adult" else "Adult"
      wink_ready = False
      print("FaceNav profile:", profile_name)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

# DOWNLOAD MEDIAPIPE MODEL
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
  urllib.request.urlretrieve(model_url, model_path)

# FACE DETECTORS
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
system_name = platform.system()

if system_name == "Windows":
  haar_folder = r"C:\FaceNavAI"
  os.makedirs(haar_folder, exist_ok=True)
  haar_path = os.path.join(haar_folder, "haarcascade_frontalface_default.xml")

  if not os.path.isfile(haar_path):
    local_haar = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

    if os.path.isfile(local_haar):
      with open(local_haar, "rb") as src:
        with open(haar_path, "wb") as dst:
          dst.write(src.read())
    else:
      haar_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
      urllib.request.urlretrieve(haar_url, haar_path)

  face_detector = cv2.CascadeClassifier(haar_path)

else:
  face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  )

if face_detector.empty():
  raise RuntimeError("Cannot load Haar face detector")

options = mp.tasks.vision.FaceLandmarkerOptions( base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                                                 running_mode=mp.tasks.vision.RunningMode.VIDEO,
                                                 num_faces=1 )

# FACIAL LANDMARK NUMBERS
points = { "Nose": 1, "NoseBridge": 168, "LeftMouth": 61, "RightMouth": 291, "OuterUpperLip": 0, "UpperLip": 13, "LowerLip": 14, 
           "Chin": 152, "LeftEyeUpper": 159, "LeftEyeLower": 145, "RightEyeUpper": 386, "RightEyeLower": 374 }

# LANDMARK COLOURS (BGR)
landmark_colors = { "Nose": (0, 255, 0), "NoseBridge": (0, 180, 0), "LeftMouth": (255, 0, 0), "RightMouth": (255, 0, 0),
                    "OuterUpperLip": (255, 255, 0), "UpperLip": (0, 0, 255), "LowerLip": (0, 0, 255), 
                    "Chin": (0, 255, 255), "LeftEyeUpper": (255, 0, 255), "LeftEyeLower": (255, 0, 255), 
                    "RightEyeUpper": (255, 0, 255), "RightEyeLower": (255, 0, 255) }

# SHORT LANDMARK LABELS
point_labels = { "Nose": "N", "NoseBridge": "NB", "LeftMouth": "LM", "RightMouth": "RM", "OuterUpperLip": "OUL", "UpperLip": "UL", 
                 "LowerLip": "LL", "Chin": "C", "LeftEyeUpper": "LEU", "LeftEyeLower": "LEL", 
                 "RightEyeUpper": "REU", "RightEyeLower": "REL" }

# CSV FIELDS
measurements = [ "LeftEyeOpening", "RightEyeOpening", "MouthOpening", "Nose_UpperLips", "MouthCenterX" ]
headers = ["Video", "Frame", "TimeSeconds", "HeadTiltDegrees"] + measurements + [ "LeftEye", "RightEye", "Mouth", "Lips", 
                                                                                  "MouseAction", "Profile" ]

# CLASSIFICATION: AVERAGE FIVE FRAMES
def process_group(group):
  valid = [row for row in group if row.get("LeftEyeOpening") is not None]

  if valid:
    averages = {name: sum(row[name] for row in valid) / len(valid) for name in measurements}
    t = profiles[profile_name]
    left_eye  = "OPEN"   if averages["LeftEyeOpening"]  >= t["left_eye_open"]  else "CLOSED"
    right_eye = "OPEN"   if averages["RightEyeOpening"] >= t["right_eye_open"] else "CLOSED"
    mouth     = "CLOSED" if averages["MouthOpening"]    <= t["mouth_open"]     else "OPEN"

    nose_lips = averages["Nose_UpperLips"]
    mouth_x = averages["MouthCenterX"]

    # Mouth OPEN disables lip directions
    if mouth == "OPEN":
      lips = "DISABLED"
    # Priority: NEUTRAL -> NOSE -> LEFT -> RIGHT -> CHIN
    elif t["nose_min"] <= nose_lips <= t["nose_max"] and t["mouth_x_min"] <= mouth_x <= t["mouth_x_max"]:
      lips = "Neutral"
    elif nose_lips < t["nose_min"]:
      lips = "Nose(Up)"
    elif mouth_x < t["mouth_x_min"]:
      lips = "Left"
    elif mouth_x > t["mouth_x_max"]:
      lips = "Right"
    elif nose_lips > t["nose_max"]:
      lips = "Chin(Down)"
    else:
      lips = "Neutral"

    for row in group:
      row.update(averages)
      row.update({"LeftEye": left_eye, "RightEye": right_eye, "Mouth": mouth, "Lips": lips})
  else:
    left_eye, right_eye, mouth, lips = "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"
    for row in group:
      row.update({"LeftEye": left_eye, "RightEye": right_eye, "Mouth": mouth, "Lips": lips})

  return left_eye, right_eye, mouth, lips

# MOUSE CONTROL - ONE ACTION PER FIVE-FRAME GROUP
def control_mouse(status):
  global mouse_enabled, wink_ready

  if not mouse_enabled:
    return "PAUSED"

  left_eye, right_eye, mouth, lips = status
  action = "None"

  try:
    # A new wink is allowed only after both eyes have been OPEN
    if left_eye == "OPEN" and right_eye == "OPEN":
      wink_ready = True

    elif wink_ready and left_eye == "CLOSED" and right_eye == "OPEN":
      pyautogui.click(button="left")
      wink_ready = False
      action = "LEFT CLICK"

    elif wink_ready and right_eye == "CLOSED" and left_eye == "OPEN":
      pyautogui.click(button="right")
      wink_ready = False
      action = "RIGHT CLICK"

    elif left_eye == "UNKNOWN" or right_eye == "UNKNOWN":
      wink_ready = False

    # Move the pointer only if the mouth is CLOSED
    if mouth == "CLOSED":
      movement = { "Left": (-mouse_step, 0), "Right": (mouse_step, 0), "Nose(Up)": (0, -mouse_step), "Chin(Down)": (0, mouse_step) }

      if lips in movement:
        dx, dy = movement[lips]
        pyautogui.moveRel(dx, dy, duration=0)
        if action == "None":
          action = "MOVE " + lips

  except pyautogui.FailSafeException:
    mouse_enabled = False
    wink_ready = False
    action = "SAFETY STOP"
    print("PyAutoGUI safety stop. Mouse control disabled.")

  return action

# HEAD-TILT CORRECTION WITHOUT ROTATING THE PHOTO
def correct_head_tilt(landmarks, width, height):
  eye_a = ((landmarks[33].x  + landmarks[133].x) * width / 2, (landmarks[33].y  + landmarks[133].y) * height / 2)
  eye_b = ((landmarks[362].x + landmarks[263].x) * width / 2, (landmarks[362].y + landmarks[263].y) * height / 2)

  left_eye, right_eye = sorted([eye_a, eye_b], key=lambda p: p[0])
  angle               = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
  eye_midpoint        = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
  matrix              = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)

  display_coordinates = {}
  corrected_coordinates = {}

  for name, number in points.items():
    x = landmarks[number].x * width
    y = landmarks[number].y * height
    display_coordinates[name] = (int(x), int(y))
    corrected_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    corrected_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    corrected_coordinates[name] = (int(corrected_x), int(corrected_y))

  return display_coordinates, corrected_coordinates, angle

# DRAW LANDMARKS
def draw_landmarks(face_view, coordinates):
  connections = [ ("NoseBridge", "Nose"), ("Nose", "Chin"), ("LeftEyeUpper", "LeftEyeLower"), ("RightEyeUpper", "RightEyeLower"),
                ("OuterUpperLip", "UpperLip"), ("UpperLip", "LowerLip"), ("LeftMouth", "RightMouth") ]

  for start, end in connections:
    if start in coordinates and end in coordinates:
      cv2.line(face_view, coordinates[start], coordinates[end], (0, 255, 255), 1)

  for name, (x, y) in coordinates.items():
    if not (0 <= x < face_view.shape[1] and 0 <= y < face_view.shape[0]):
      continue

    color = landmark_colors[name]
    cv2.circle(face_view, (x, y), 5, (0, 0, 0), -1)
    cv2.circle(face_view, (x, y), 3, color, -1)
    label_x = min(max(x + 6, 0), face_view.shape[1] - 32)
    label_y = min(max(y - 6, 12), face_view.shape[0] - 4)
    cv2.putText(face_view, point_labels[name], (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
    cv2.putText(face_view, point_labels[name], (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

  return face_view

# CREATE COLOURED LIVE DISPLAY
def create_display(gray_face, coordinates, status, elapsed, angle, enabled, action):
  face_view = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
  if coordinates:
    face_view = draw_landmarks(face_view, coordinates)

  panel = np.full((310, 390, 3), (40, 40, 40), dtype=np.uint8)
  colors = { "OPEN": (0, 255, 0), "CLOSED": (0, 0, 255), "Nose(Up)": (255, 220, 100), "Chin(Down)": (66, 113, 169),
             "Left": (0, 165, 255), "Right": (200, 105, 255), "Neutral": (255, 255, 255), "DISABLED": (140, 140, 140),
             "UNKNOWN": (160, 160, 160) }

  minutes, seconds = divmod(int(elapsed), 60)
  cv2.putText(panel, "FaceNav AI - LIVE",                  (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
  cv2.putText(panel, f"Time: {minutes:02d}:{seconds:02d}", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

  labels = ["Left Eye", "Right Eye", "Mouth", "Lips"]
  for i, label in enumerate(labels):
    value = status[i]
    y = 108 + i * 39
    cv2.putText(panel, label + ":", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(panel, value, (145, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors.get(value, (255, 255, 255)), 2)

  control_color = (0, 255, 0) if enabled else (0, 165, 255)
  control_text = "ON" if enabled else "OFF"
  cv2.putText(panel, "Mouse Control: " + control_text, (15, 237),
              cv2.FONT_HERSHEY_SIMPLEX, 0.52, control_color, 2)
  cv2.putText(panel, "Action: " + action[:30], (15, 259),
              cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
  cv2.putText(panel, f"Tilt: {angle:+.1f} deg", (205, 68),
              cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220, 220, 220), 1)
  cv2.rectangle(panel, (135, 270), (255, 299), (90, 90, 90), -1)
  cv2.rectangle(panel, (135, 270), (255, 299), (200, 200, 200), 1)
  cv2.putText(panel, profile_name.upper(), (147, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
  return np.hstack((face_view, panel))

# OPEN WEBCAM
camera = cv2.VideoCapture(camera_number)
if not camera.isOpened():
  raise RuntimeError("Cannot open webcam. Check macOS camera permissions.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

detector = None

try:
  detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(window_name, 1035, 465)
  cv2.setMouseCallback(window_name, on_display_click)

  print("FaceNav AI live camera started")
  print("Target FPS:", target_fps)
  print("Averaging:", average_frames, "frames")
  print("Chin extension:", chin_extension)
  print("Mouse control starts OFF. Open mouth to toggle ON/OFF.")
  print("Close both eyes to hide/show the entire FaceNav AI window.")
  print("Click ADULT/CHILD inside the window to change thresholds.")
  print("When hidden, use both eyes CLOSED again after reopening to restore the window.")
  print("When hidden, press Ctrl+C in Terminal to quit.")
  print("Press Q or ESC to stop.")

  frame_number = 0
  group = []
  status = ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN")
  gray_face = np.full((310, 300), 255, dtype=np.uint8)
  display_coordinates = {}
  head_angle = 0.0
  start_time = time.monotonic()
  next_sample_time = start_time
  last_timestamp_ms = -1

  with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()

    while True:
      key = cv2.waitKey(1) & 0xFF if window_enabled else 255

      if key in (ord("q"), ord("Q"), 27):
        break

      if window_enabled:
        try:
          if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        except cv2.error:
          break

      success, frame = camera.read()
      if not success:
        print("Cannot read webcam frame")
        break

      if mirror_camera:
        frame = cv2.flip(frame, 1)

      now = time.monotonic()
      elapsed = now - start_time

      if now >= next_sample_time:
        next_sample_time = now + 1 / target_fps
        frame_number += 1
        timestamp_ms = max(int(elapsed * 1000), last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms

        faces = face_detector.detectMultiScale(frame, 1.1, 8)
        gray_face = np.full((310, 300), 255, dtype=np.uint8)
        display_coordinates = {}
        head_angle = 0.0
        face_found = False

        if len(faces) > 0:
          x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
          cropped_face = frame[y:min(y+h+chin_extension, frame.shape[0]), x:min(x+w, frame.shape[1])]

          if cropped_face.size > 0:
            resized_face = cv2.resize(cropped_face, (300, 310))
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            face_found = True

        row = {"Video": "LiveCamera", "Frame": frame_number, "TimeSeconds": round(elapsed, 3)}

        # MEDIAPIPE LANDMARK ANALYSIS
        if face_found:
          rgb = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB)
          image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
          result = detector.detect_for_video(image, timestamp_ms)

          if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            height, width = gray_face.shape[:2]
            display_coordinates, coordinates, head_angle = correct_head_tilt(landmarks, width, height)
            row["HeadTiltDegrees"] = round(head_angle, 2)

            left_eye       = abs(coordinates["LeftEyeLower"][1]  - coordinates["LeftEyeUpper"][1] )
            right_eye      = abs(coordinates["RightEyeLower"][1] - coordinates["RightEyeUpper"][1])
            lip_distance   = abs(coordinates["UpperLip"][1]      - coordinates["LowerLip"][1]     )
            nose_upperlips = abs(coordinates["OuterUpperLip"][1] - coordinates["Nose"][1]         )

            mouth_center_x =    (coordinates["LeftMouth"][0]     + coordinates["RightMouth"][0]   ) / 2
            mouth_center_relative = mouth_center_x - coordinates["Nose"][0]

            row.update({ "LeftEyeOpening": left_eye, "RightEyeOpening": right_eye, "MouthOpening": lip_distance,
              "Nose_UpperLips": nose_upperlips, "MouthCenterX": mouth_center_relative})

        group.append(row)
           
        # Once we collected 5 Frames we will Average them and update the CSV file.
        if len(group) == average_frames:
          status = process_group(group)
          left_state, right_state, mouth_state = status[:3]
          mouse_just_toggled = False

          # OPEN mouth toggles mouse control once; CLOSED mouth resets the toggle
          if mouth_state == "CLOSED":
            mouth_toggle_ready = True
          elif mouth_state == "OPEN" and mouth_toggle_ready:
            mouse_enabled = not mouse_enabled
            mouth_toggle_ready = False
            mouse_just_toggled = True
            wink_ready = False
            print("Mouse control:", "ON" if mouse_enabled else "OFF")

          # Both eyes CLOSED toggle the entire window; both OPEN reset the toggle
          if left_state == "CLOSED" and right_state == "CLOSED":
            wink_ready = False
          if left_state == "OPEN" and right_state == "OPEN":
            eyes_toggle_ready = True
          elif left_state == "CLOSED" and right_state == "CLOSED" and eyes_toggle_ready:
            window_enabled = not window_enabled
            eyes_toggle_ready = False
            if window_enabled:
              cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
              cv2.resizeWindow(window_name, 1035, 465)
              cv2.setMouseCallback(window_name, on_display_click)
            else:
              cv2.destroyWindow(window_name)
            print("Display window:", "ON" if window_enabled else "OFF")

          # Avoid moving or clicking in the same group used to toggle the mouse
          if mouse_just_toggled:
            last_mouse_action = "ENABLED" if mouse_enabled else "PAUSED"
          else:
            last_mouse_action = control_mouse(status)

          # Record the mouse action only on the group's final frame
          for item in group:
            item["MouseAction"] = ""
            item["Profile"] = profile_name
          group[-1]["MouseAction"] = last_mouse_action

          writer.writerows(group)
          file.flush()
          group = []

      if window_enabled:
        display = create_display(gray_face, display_coordinates, status, elapsed,
                                 head_angle, mouse_enabled, last_mouse_action)
        cv2.imshow(window_name, display)
      else:
        time.sleep(0.01)

    # Save remaining frames without triggering a mouse action on exit
    if group:
      status = process_group(group)
      for item in group:
        item["MouseAction"] = "STOPPED"
        item["Profile"] = profile_name
      writer.writerows(group)
      file.flush()

except KeyboardInterrupt:
  print("Stopped using Ctrl+C")

finally:
  print("Releasing camera and closing MediaPipe...")
  camera.release()
  if detector is not None:
    detector.close()
  cv2.destroyAllWindows()
  print("Live camera stopped")
  print("CSV saved to:", os.path.abspath(output_file))
