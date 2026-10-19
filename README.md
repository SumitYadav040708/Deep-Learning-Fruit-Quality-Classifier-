# Fruit Quality Analyzer

A Streamlit application that runs four trained YOLO11 models on each uploaded fruit image. It shows each model's independent predictions, detection table, and annotated image.

## Model Capabilities and Limitations

The app runs all four checkpoints on every analyzed image:

- `det_n_s0.pt`: fruit detection (bounding boxes; no masks).
- `seg_n_s0.pt`: fruit instance segmentation.
- `seg_cbam_n_s0.pt`: CBAM fruit instance segmentation.
- `seg_cbam_focal_dice_n_s0.pt`: CBAM fruit instance segmentation trained with focal/Dice mask loss.

All four models predict the same six fruit-level classes: `apple`, `banana`, `orange`, `rotten_apple`, `rotten_banana`, and `rotten_orange`. Each result tab displays only that checkpoint's predictions. The three segmentation models produce whole-fruit masks; none of the models segments rotten patches or predicts a rotten percentage. Detection confidence is not the amount of rot, and neither a whole-fruit mask nor a bounding box establishes a rotten percentage. These labels are model predictions, not verified judgments of food safety.

The CBAM checkpoints use Ultralytics' built-in `ultralytics.nn.modules.conv.CBAM`. The app pins the recorded Ultralytics version and reports a clear error if that module cannot be imported. Focal/Dice loss was a training-time change; no custom loss module is needed for inference.

## Project Structure

```text
fruit-quality-app/
|-- app.py
|-- det_n_s0.pt
|-- seg_n_s0.pt
|-- seg_cbam_n_s0.pt
|-- seg_cbam_focal_dice_n_s0.pt
|-- data_seg_trainval.yaml
|-- train_config.json
|-- requirements.txt
`-- README.md
```

Keep all four `.pt` checkpoints beside `app.py`; the app requires all four and does not substitute a pretrained model. `data_seg_trainval.yaml` describes the training/validation dataset; it is not needed to run inference. `train_config.json` records training and architecture metadata. The app does not train or modify the models.

## Windows Setup

Use Python 3.10 or newer with a compatible PyTorch build. From PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Ultralytics depends on PyTorch and torchvision. The requirements file does not force a CUDA-specific build; use a CPU-compatible PyTorch installation if pip does not resolve one suitable for your machine. For a deliberate CPU-only setup, install the appropriate CPU wheels from the official PyTorch installation instructions, then install the project requirements. CUDA is optional.

## Analyze an Image

1. Start the app with the command above and open the local Streamlit URL shown in the terminal.
2. In the sidebar, choose a confidence threshold (default `0.25`) and inference image size (default `640`). Whole-fruit masks and bounding boxes/labels can be toggled independently.
3. Upload one JPG, JPEG, PNG, or WEBP image and select **Analyze with all four models**.
4. Review each model's independent annotated image, summary metrics, and per-detection table in its own tab. Download that model's table as CSV or annotated image as PNG.

Uploads and exports are kept in memory for the session; the app does not save uploaded images to disk.

## Interpreting Results

Each result tab uses that checkpoint's class names and confidence values. `apple`, `banana`, and `orange` count as fresh-class predictions; `rotten_apple`, `rotten_banana`, and `rotten_orange` count as rotten-class predictions. These are model outputs and do not establish food safety. Confidence reflects the model's confidence in a detection, not a percentage of rotten tissue. Coordinates are pixel positions in the uploaded image. A no-detection result is valid and produces an empty CSV with the table headers.

Only the three segmentation model tabs show whole-fruit masks. Rotten percentages are intentionally not displayed.

## Troubleshooting

- **A checkpoint is missing:** Put all four named `.pt` files in the project directory beside `app.py`. The app never substitutes a pretrained model.
- **Missing packages:** Activate `.venv` and run `python -m pip install -r requirements.txt`. Confirm the selected VS Code interpreter is `.venv`.
- **CBAM or checkpoint loading error:** The checkpoint refers to `ultralytics.nn.modules.conv.CBAM`. Confirm `ultralytics==8.4.177` is installed in the active environment (`python -m pip show ultralytics`) and that the package imports (`python -c "from ultralytics.nn.modules.conv import CBAM; print(CBAM)"`). If that exact module is unavailable in the installed package, share the full terminal traceback and training environment's Ultralytics source/module implementation; this repository does not contain a separate custom CBAM implementation to substitute.
- **PyTorch/torchvision import or binary error:** Install mutually compatible PyTorch and torchvision builds for the selected Python version and CPU/CUDA target, then rerun `python -m pip install -r requirements.txt`.
- **No detections:** Try a lower confidence threshold, check that the image clearly contains fruit, and confirm the checkpoints are the expected project models.
- **Invalid upload or inference error:** Try a readable JPG, PNG, or WEBP image. The terminal logs include diagnostic tracebacks for image, checkpoint, and inference failures.
