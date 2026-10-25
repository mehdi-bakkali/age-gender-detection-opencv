# Age & Gender Estimation with OpenCV

A compact computer-vision project that detects faces and predicts an approximate age range and a model gender label from a photo, video, or webcam stream. It uses OpenCV's DNN module with pretrained TensorFlow and Caffe models.

> **Demo note:** Predictions are estimates from pretrained models. They may be inaccurate and should not be treated as facts about a person's age or gender, or used to make consequential decisions.

## Features

- Detects one or more faces in each frame.
- Predicts one of eight age ranges and one of two model labels (`Male` or `Female`).
- Supports still images, video files, and live webcam input.
- Displays model confidence scores and can save an annotated image.
- Downloads model files on demand; large weights are kept out of Git.

## Quick start

### Requirements

- Python 3.10–3.14
- OpenCV 4 (OpenCV 5 removed the Caffe importer required by these models)
- A desktop session for the image and webcam preview windows

### Install

```bash
git clone https://github.com/mehdi-bakkali/age-gender-detection-opencv.git
cd age-gender-detection-opencv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_models.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The model download step fetches six configuration and weight files into `models/`. The Caffe weight files are large, so the initial download may take a little while.

## Usage

### Analyze a photo

```bash
python gad.py --image path/to/photo.jpg
```

Save the annotated image as well as previewing it:

```bash
python gad.py --image path/to/photo.jpg --output outputs/result.jpg
```

### Analyze a video

```bash
python gad.py --video path/to/video.mp4
```

### Use a webcam

```bash
python gad.py --camera 0
```

Press **Q** or **Esc** to close the video window. Change `0` if your camera uses a different device index.

## How it works

1. An OpenCV DNN face detector finds face bounding boxes in the input frame.
2. Each detected face is cropped, resized to 227 × 227 pixels, and mean-subtracted.
3. Separate pretrained Caffe networks classify the crop into a gender label and an age range.
4. OpenCV draws the predictions and confidence scores over the frame.

The available age ranges are `0–2`, `4–6`, `8–12`, `15–20`, `25–32`, `38–43`, `48–53`, and `60–100`. The age model was trained as a classifier, so it does not return an exact age.


## Troubleshooting

- **Missing model files:** Run `python download_models.py` from the project directory.
- **`No module named cv2`:** Activate `.venv` and run `python -m pip install -r requirements.txt`.
- **OpenCV Caffe importer error:** Confirm OpenCV 4 is installed with `python -c "import cv2; print(cv2.__version__)"`. Reinstall the pinned dependency with `python -m pip install --force-reinstall -r requirements.txt`.
- **No face detected:** Try a clear, well-lit image with a larger, mostly front-facing face.
- **Preview window does not open:** OpenCV display windows require a desktop session. Use `--output` for image inference in a headless environment.

## Credits

- Project walkthrough: [DataFlair – Gender and Age Detection with OpenCV](https://data-flair.training/blogs/python-project-gender-age-detection/)
- Pretrained age and gender models: Gil Levi and Tal Hassner, [Age and Gender Classification Using Convolutional Neural Networks](https://talhassner.github.io/home/publication/2015_CVPR)
- Face detector and OpenCV DNN example: [OpenCV / LearnOpenCV AgeGender project](https://github.com/spmallick/learnopencv/tree/master/AgeGender)

Model files are downloaded from their respective source repositories at setup time and are not included in this project. Please review the original model and dataset terms before redistribution or commercial use.
