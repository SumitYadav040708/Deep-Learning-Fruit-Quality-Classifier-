# Fruit Quality Analyzer

A Streamlit application for running the project's trained YOLO11 instance-segmentation checkpoint on fruit images. It displays the original image alongside whole-fruit masks and detections, reports the model's class and confidence predictions, and exports a detection table and annotated PNG.

## Model Capabilities and Limitations

The `best.pt` checkpoint predicts six fruit-level classes: `apple`, `banana`, `orange`, `rotten_apple`, `rotten_banana`, and `rotten_orange`. Its segmentation masks cover whole fruit instances. The labels are predictions, not verified judgments of food safety.

This model does **not** segment rotten patches or predict a rotten percentage. Detection confidence is not the amount of rot, and neither the whole-fruit mask nor its bounding box can establish a rotten percentage. Pixel-level rotten-region analysis requires separately annotated rotten-region masks and a model trained on those annotations.

Training metadata records Ultralytics 8.4.177 and an architecture using Ultralytics' built-in `ultralytics.nn.modules.conv.CBAM` layers. The checkpoint serializes references to that built-in module. Focal/Dice loss was a training-time loss change; the checkpoint contains no custom loss module needed for inference. No custom CBAM source file was present in this project. The app pins the recorded Ultralytics version and reports a clear error if its CBAM module cannot be imported.

## Project Structure

```text
fruit-quality-app/
|-- app.py
|-- best.pt
|-- data_seg_trainval.yaml
|-- train_config.json
|-- requirements.txt
`-- README.md
```

`best.pt` is the trained checkpoint. `data_seg_trainval.yaml` describes the training/validation dataset; it is not needed to run inference. `train_config.json` records training and architecture metadata. The app does not train or modify the model.

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
3. Upload one JPG, JPEG, PNG, or WEBP image and select **Analyze image**.
4. Review the original and annotated images, summary metrics, and per-detection table. Download the table as CSV or the annotated image as PNG.

Uploads and exports are kept in memory for the session; the app does not save uploaded images to disk.

## Interpreting Results

The model's checkpoint class names and predicted confidence values are used in the results. `apple`, `banana`, and `orange` count as fresh-class predictions; `rotten_apple`, `rotten_banana`, and `rotten_orange` count as rotten-class predictions. These are model outputs and do not establish food safety. Confidence reflects the model's confidence in a detection, not a percentage of rotten tissue. Coordinates are pixel positions in the uploaded image. A no-detection result is valid and produces an empty CSV with the table headers.

The masks shown in the app delineate whole fruit instances only. Rotten percentages are intentionally not displayed. Future rotten-region segmentation can be added as a separate prediction source without changing the fruit detection table and export flow.

## Troubleshooting

- **`best.pt` is missing:** Put the original checkpoint in the project directory beside `app.py`. The app never substitutes a pretrained model.
- **Missing packages:** Activate `.venv` and run `python -m pip install -r requirements.txt`. Confirm the selected VS Code interpreter is `.venv`.
- **CBAM or checkpoint loading error:** The checkpoint refers to `ultralytics.nn.modules.conv.CBAM`. Confirm `ultralytics==8.4.177` is installed in the active environment (`python -m pip show ultralytics`) and that the package imports (`python -c "from ultralytics.nn.modules.conv import CBAM; print(CBAM)"`). If that exact module is unavailable in the installed package, share the full terminal traceback and training environment's Ultralytics source/module implementation; this repository does not contain a separate custom CBAM implementation to substitute.
- **PyTorch/torchvision import or binary error:** Install mutually compatible PyTorch and torchvision builds for the selected Python version and CPU/CUDA target, then rerun `python -m pip install -r requirements.txt`.
- **No detections:** Try a lower confidence threshold, check that the image clearly contains fruit, and confirm the correct `best.pt` is beside the app.
- **Invalid upload or inference error:** Try a readable JPG, PNG, or WEBP image. The terminal logs include diagnostic tracebacks for image, checkpoint, and inference failures.
