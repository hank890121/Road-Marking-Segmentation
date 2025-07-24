## Environment

**System Software**:

- OS: Ubuntu 20.04 LTS
- CUDA：12.3
- CuDNN：8.7.0.0
- Python：3.11

**Installation**:

Install pytorch-2.4.0 and torchvision-0.19.0.
Then install the requirements with
```
pip install -r requirements.txt
```

## Inference

To inference a folder of images:
```
python -m tools.inference_folder <path/to/your/category/csv> <path/to/your/images>\
 <suffix> <path/to/your/output> <path/to/your/checkpoint> <height> <width> <use _sliding_inference>
```
For example,
```
python -m tools.inference_folder data/csv/rlmd.csv data/rlmd_ac/clear/val/images\
 .jpg inference_output weight/80000.pth 16 1080 1920--sliding-window
```

To inference a video:
```
python -m tools.inference_video <path/to/your/category/csv> <path/to/your/video>\
 <path/to/your/output> <path/to/your/checkpoint> <height> <width> <output_framerate> <use _sliding_inference>
```
For example,
```
python -m tools.inference_video data/csv/rlmd.csv sample_video.mp4\
inference_output.mp4 weight/80000.pth 1080 1920 30 --sliding-window
```
