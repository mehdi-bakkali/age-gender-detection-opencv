"""Download the pretrained OpenCV age and gender model files."""

from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
BASE = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/"
FILES = {
    "opencv_face_detector.pbtxt": BASE + "opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": BASE + "opencv_face_detector_uint8.pb",
    "age_deploy.prototxt": BASE + "age_deploy.prototxt",
    "age_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    "gender_deploy.prototxt": BASE + "gender_deploy.prototxt",
    "gender_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
}


def download(name: str, url: str) -> None:
    destination = MODEL_DIR / name
    if destination.exists() and destination.stat().st_size > 1024:
        print(f"Already present: {name}")
        return
    print(f"Downloading {name}...")
    request = Request(url, headers={"User-Agent": "Python age-gender-demo"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size < 1024:
            raise RuntimeError(f"Download for {name} was unexpectedly small")
        if temporary.read_bytes()[:100].lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise RuntimeError(f"The server returned a web page instead of {name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for filename, url in FILES.items():
        download(filename, url)
    print(f"\nModels are ready in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
