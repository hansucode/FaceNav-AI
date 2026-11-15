# FaceNav AI

**FaceNav AI** is an experimental local webcam-based facial gesture controller written in Python. It uses OpenCV to crop the face, Google's MediaPipe Face Landmarker to measure facial landmarks, and PyAutoGUI to move and click the mouse.

**Development status:** The code was developed on macOS. Windows is expected to be compatible with the libraries, but has not been tested by the developer. Results and webcam behaviour depend on hardware, lighting, operating-system permissions and your environment. This is an experimental accessibility prototype, not a validated assistive technology product.

## What it does

- Lip movements LEFT / RIGHT / NOSE (up) / CHIN (down) move the cursor.
- Left-eye wink triggers left click; right-eye wink triggers right click (verify left/right labels on your own camera).
- Opening the mouth toggles mouse control. Mouse control starts **OFF**.
- Closing both eyes toggles the entire OpenCV display window. Webcam tracking continues when the window is hidden.
- A clickable ADULT / CHILD switch selects separately configurable thresholds.
- Face detection, landmark plotting, angle-based head-tilt compensation (without rotating the visible photo), averaging over five processed frames, and CSV logging are included.

## Requirements

- Windows or macOS computer with a webcam and a desktop session.
- **Python 3.11 recommended** for a new installation. The author's older Python 3.8 environment may require older package versions; using a fresh Python 3.11 environment is simpler. Verify OS/Python compatibility of current MediaPipe and PyAutoGUI releases for your machine.
- Internet access on **first launch**, to download Google's `face_landmarker.task` model (then cached locally).
- Webcam permission; macOS also requires **Accessibility** permission for the app running Python to control the mouse. Windows webcam privacy settings and organisation security policies can also apply.

## Download the project

Click GitHub **Code → Download ZIP**, extract it, and open Terminal (macOS) or PowerShell (Windows) in the extracted `FaceNav-AI` directory.

Or, if you have Git installed:

```bash
git clone https://github.com/YOUR-USERNAME/FaceNav-AI.git
cd FaceNav-AI
```

Replace `YOUR-USERNAME` after publishing the repository.

## Installation — Windows (PowerShell)

Install Python 3.11 from https://www.python.org/downloads/ and enable the Python launcher. Then, from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe facenav.py
```

Using the virtual environment's Python directly avoids PowerShell's execution policy restrictions. To use the notebook instead:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Then open `FaceNav_AI.ipynb` in the browser. The live camera appears in a separate **OpenCV desktop window**, not embedded in the notebook.

## Installation — macOS (Terminal)

Install Python 3.11 if it is not already installed. From the project folder:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python facenav.py
```

For the notebook:

```bash
./.venv/bin/python -m notebook
```

On macOS, check **System Settings → Privacy & Security → Camera** and **Accessibility** for Terminal, Python, Jupyter or the app launching Python. You may need to restart that app after changing permissions.

## How to use

1. Sit in good lighting with one face visible to the webcam.
2. The default profile in `facenav.py` is **Child**. Click the ADULT/CHILD switch within the OpenCV window to change thresholds.
3. Close your mouth before your first deliberate mouth-opening gesture. Opening your mouth toggles mouse control ON/OFF (one toggle per open/close cycle).
4. Move the lips left/right/up/down to steer the cursor; wink one eye while keeping the other open to click.
5. Close both eyes to hide/show the **entire** OpenCV window. When hidden, it continues tracking; use Ctrl+C in the terminal to terminate if needed.
6. Press `Q` or `Esc` while the OpenCV window has focus to stop. Close the application before opening the generated CSV in another program.

**Safety:** Mouse actions can activate other windows. Keep a physical input device available. PyAutoGUI's upper-left-corner failsafe is enabled. If eye or mouth recognition is unreliable, change `profiles` thresholds in the source code based on your own measurements.

## Adjusting settings

At the top of `facenav.py`, change `camera_number`, `target_fps`, `average_frames`, `chin_extension`, `mirror_camera`, and `mouse_step` as needed. Thresholds for **Adult** and **Child** are stored separately in the `profiles` dictionary. The provided profile values are experimental and not universal adult/child measurements.

## Outputs and privacy

The program writes `mediapipe_live_output.csv` to your project folder and downloads `face_landmarker.task` on first launch. Both are ignored by Git via `.gitignore`. **Do not commit personal face photos/videos or facial measurement CSVs** without informed consent, especially for minors. Camera frames are processed locally by this script; the model file is downloaded from Google's hosting on first use.

## Troubleshooting

- **No camera:** Close other webcam applications; check camera permissions; try `camera_number = 1`.
- **No mouse movement:** First confirm that a mouth-open gesture toggles `Mouse Control: ON`; check macOS Accessibility permission; test PyAutoGUI separately.
- **PyAutoGUI installation complains about PyObjC needing Python >=3.9:** You are likely installing into Python 3.8. Create a fresh Python 3.11 virtual environment.
- **Window freezes or will not close:** Run the program as `facenav.py` from Terminal/PowerShell, not inside a notebook; try `Ctrl+C` in the terminal. OpenCV desktop GUI behaviour depends on the environment.
- **Left and right reversed:** Check physical-eye versus camera-mirror labels before using clicks. Calibration may differ on another webcam.

## Licence

No licence is included yet. If you want others to reuse or modify the project, add an appropriate open-source licence (for example MIT), after checking the licences of the dependencies and any competition requirements.
