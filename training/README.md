# Custom model training

The `custom_training.ipynb` notebook is designed for Google Colab and walks
through training a lightweight Ultralytics YOLO model without embedding API
keys. It supports an uploaded YOLO-format dataset or a dataset already exported
from Roboflow.

## Expected dataset layout

```text
dataset/
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/
|   |-- images/
|   `-- labels/
|-- test/                 # optional
`-- dataset.yaml
```

Each label line uses normalized YOLO format:
`class_id x_center y_center width height`.

After training, download `runs/detect/smartvision-lite/weights/best.pt`. Do not
commit it. Put it in the local `models/` directory and select it from the
Streamlit sidebar.
