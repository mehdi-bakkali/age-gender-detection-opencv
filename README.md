# Approximate age-range and gender prediction

This project follows the DataFlair OpenCV DNN example. It detects faces and runs pretrained classifiers that estimate one of eight age ranges and one of two labels (`Male` or `Female`). These are model categories, not reliable statements about a person's actual age or gender.

## 1. Set up Python

Python 3.10–3.14 is recommended. This project needs OpenCV 4 because OpenCV 5 removed the Caffe model importer used by the pretrained age and gender models. From this project folder, create and activate a virtual environment, then install OpenCV:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate` instead of the `source` command.

## 2. Download the pretrained models

```bash
python download_models.py
```

The six model/configuration files go into `models/`. The classifier weights are large downloads; a connection is needed for this step.

## 3. Run it

Use an image (the result opens in a window):

```bash
python gad.py --image path/to/photo.jpg
```

Save the annotated image too:

```bash
python gad.py --image path/to/photo.jpg --output outputs/result.jpg
```

Use a webcam, press **Q** or **Esc** to stop:

```bash
python gad.py --camera 0
```

Analyze a video file:

```bash
python gad.py --video path/to/video.mp4
```

## How the pipeline works

1. The TensorFlow face detector returns face boxes and confidence scores.
2. Each face crop is resized to 227×227 and mean-subtracted for the Caffe classifiers.
3. The highest scoring gender and age-range classes are drawn over the image.

The age labels are `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, and `(60-100)`. The original models were trained on the Adience dataset and can make inaccurate predictions, especially for faces unlike their training examples. Use respectfully and avoid consequential decisions based on these outputs.

## Troubleshooting

- **Missing model files:** run `python download_models.py` from this folder.
- **`No module named cv2`:** activate the virtual environment and run `python -m pip install -r requirements.txt`.
- **No face found:** use a clear, front-facing, well-lit image with a reasonably large face.
- **GUI display error:** OpenCV's image/video windows need a desktop session; use `--output` for image processing where a window is unavailable.
