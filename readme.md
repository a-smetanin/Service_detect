## Usage

To upload and process a video file, use the following `curl` command:

```bash
curl -X POST http://localhost:5000/upload \
  -F "video=@1.mp4" \
  -F "model_type=all"
```
# Model Type Options
* all: Use both YOLOv5 and YOLOv8 models.
* yolov5: Use only the YOLOv5 model.
* yolov8: Use only the YOLOv8 model.