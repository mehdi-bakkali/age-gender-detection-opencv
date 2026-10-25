"""Approximate age-range and model-class gender prediction with OpenCV DNN."""

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
AGE_LABELS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
GENDER_LABELS = ["Male", "Female"]
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


def load_models():
    required = [
        "opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt",
        "age_net.caffemodel", "age_deploy.prototxt",
        "gender_net.caffemodel", "gender_deploy.prototxt",
    ]
    missing = [name for name in required if not (MODEL_DIR / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Missing model files: {names}\nRun: python download_models.py")
    face_net = cv2.dnn.readNet(str(MODEL_DIR / required[0]), str(MODEL_DIR / required[1]))
    age_net = cv2.dnn.readNet(str(MODEL_DIR / required[2]), str(MODEL_DIR / required[3]))
    gender_net = cv2.dnn.readNet(str(MODEL_DIR / required[4]), str(MODEL_DIR / required[5]))
    return face_net, age_net, gender_net


def analyze(frame, face_net, age_net, gender_net, threshold=0.7):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    results = []

    for index in range(detections.shape[2]):
        confidence = float(detections[0, 0, index, 2])
        if confidence < threshold:
            continue
        x1, y1, x2, y2 = (detections[0, 0, index, 3:7] * [width, height, width, height]).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        face_blob = cv2.dnn.blobFromImage(
            frame[y1:y2, x1:x2], 1.0, (227, 227), MODEL_MEAN, swapRB=False
        )
        gender_net.setInput(face_blob)
        gender_scores = gender_net.forward()[0]
        age_net.setInput(face_blob)
        age_scores = age_net.forward()[0]
        gender_i = int(gender_scores.argmax())
        age_i = int(age_scores.argmax())
        results.append((x1, y1, x2, y2, GENDER_LABELS[gender_i], AGE_LABELS[age_i],
                        float(gender_scores[gender_i]), float(age_scores[age_i])))
    return results


def annotate(frame, predictions):
    output = frame.copy()
    for x1, y1, x2, y2, gender, age, gender_conf, age_conf in predictions:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{gender} ({gender_conf:.0%}), age range {age} ({age_conf:.0%})"
        top = max(24, y1 - 10)
        cv2.putText(output, label, (x1, top), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 0), 2, cv2.LINE_AA)
    return output


def run_image(path: Path, output_path: Path | None, models):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    result = annotate(image, analyze(image, *models))
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), result):
            raise OSError(f"Could not write output image: {output_path}")
        print(f"Saved annotated image to {output_path}")
    cv2.imshow("Approximate age and gender predictions", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_stream(source, models):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise OSError(f"Could not open video source: {source}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = annotate(frame, analyze(frame, *models))
            cv2.imshow("Approximate age and gender predictions (press Q to quit)", result)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Analyze one image")
    parser.add_argument("--video", type=Path, help="Analyze a video file")
    parser.add_argument("--camera", type=int, help="Use a camera device (usually 0)")
    parser.add_argument("--output", type=Path, help="Save the annotated image")
    args = parser.parse_args()
    choices = sum(value is not None for value in (args.image, args.video, args.camera))
    if choices != 1:
        parser.error("choose exactly one of --image, --video, or --camera")
    models = load_models()
    if args.image:
        run_image(args.image, args.output, models)
    else:
        if args.output:
            parser.error("--output can only be used with --image")
        run_stream(str(args.video) if args.video else args.camera, models)


if __name__ == "__main__":
    main()
