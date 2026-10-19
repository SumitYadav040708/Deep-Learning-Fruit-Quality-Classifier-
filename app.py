from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger("fruit_quality_analyzer")
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS = (
	{
		"key": "det_n_s0",
		"label": "Detection",
		"filename": "det_n_s0.pt",
		"task": "detect",
	},
	{
		"key": "seg_cbam_n_s0",
		"label": "CBAM segmentation",
		"filename": "seg_cbam_n_s0.pt",
		"task": "segment",
	},
	{
		"key": "seg_n_s0",
		"label": "Segmentation",
		"filename": "seg_n_s0.pt",
		"task": "segment",
	},
	{
		"key": "seg_cbam_focal_dice_n_s0",
		"label": "CBAM + Focal/Dice segmentation",
		"filename": "seg_cbam_focal_dice_n_s0.pt",
		"task": "segment",
	},
)
FRESH_CLASSES = {"apple", "banana", "orange"}
ROTTEN_CLASSES = {"rotten_apple", "rotten_banana", "rotten_orange"}
TABLE_COLUMNS = [
	"Detection",
	"Predicted class",
	"Confidence",
	"X1",
	"Y1",
	"X2",
	"Y2",
]


st.set_page_config(
	page_title="Fruit Quality Analyzer",
	page_icon="🍎",
	layout="wide",
	initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading all four fruit models...")
def load_models() -> dict[str, Any]:
	"""Load all project checkpoints using their recorded Ultralytics architectures."""
	try:
		from ultralytics import YOLO
	except Exception as exc:
		LOGGER.exception("Unable to import Ultralytics and its runtime dependencies")
		raise RuntimeError(
			"Ultralytics or one of its dependencies is missing. Install the "
			"project requirements with `python -m pip install -r requirements.txt`."
		) from exc

	models: dict[str, Any] = {}
	for config in MODEL_CONFIGS:
		model_path = PROJECT_DIR / config["filename"]
		if not model_path.is_file():
			raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
		try:
			model = YOLO(str(model_path))
		except Exception as exc:
			LOGGER.exception("Unable to load checkpoint at %s", model_path)
			module_name = getattr(exc, "name", None)
			if module_name == "ultralytics.nn.modules.conv" or "CBAM" in str(exc):
				raise RuntimeError(
					f"{config['filename']} requires Ultralytics' built-in CBAM module "
					"(`ultralytics.nn.modules.conv.CBAM`). Install the pinned "
					"Ultralytics 8.4.177 version from requirements.txt."
				) from exc
			raise RuntimeError(f"Could not load {config['filename']}: {exc}") from exc
		if model.task != config["task"]:
			raise RuntimeError(
				f"{config['filename']} is a {model.task} model; "
				f"expected a {config['task']} model."
			)
		models[config["key"]] = model
	return models


def read_uploaded_image(image_bytes: bytes) -> Image.Image:
	"""Decode an upload fully so invalid and truncated images fail early."""
	with Image.open(io.BytesIO(image_bytes)) as source:
		source.load()
		return source.convert("RGB")


def class_name_for(names: Any, class_id: int) -> str:
	if isinstance(names, dict):
		return str(names.get(class_id, names.get(str(class_id), class_id)))
	if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
		return str(names[class_id])
	return str(class_id)


def build_detection_table(result: Any) -> pd.DataFrame:
	rows: list[dict[str, Any]] = []
	boxes = result.boxes
	if boxes is not None:
		coordinates = boxes.xyxy.detach().cpu().tolist()
		confidences = boxes.conf.detach().cpu().tolist()
		class_ids = boxes.cls.detach().cpu().tolist()
		for index, (box, confidence, class_id) in enumerate(
			zip(coordinates, confidences, class_ids), start=1
		):
			x1, y1, x2, y2 = box
			rows.append(
				{
					"Detection": index,
					"Predicted class": class_name_for(result.names, int(class_id)),
					"Confidence": float(confidence),
					"X1": round(float(x1), 2),
					"Y1": round(float(y1), 2),
					"X2": round(float(x2), 2),
					"Y2": round(float(y2), 2),
				}
			)
	return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def make_annotated_image(result: Any, show_masks: bool, show_boxes: bool) -> Image.Image:
	annotated_bgr = result.plot(
		masks=show_masks,
		boxes=show_boxes,
		labels=True,
		conf=True,
	)
	return Image.fromarray(np.asarray(annotated_bgr)[:, :, ::-1].copy())


def render_results(
	model_label: str,
	model_filename: str,
	analysis: dict[str, Any],
	original_image: Image.Image,
	show_masks: bool,
	show_boxes: bool,
	supports_masks: bool,
	confidence_threshold: float,
	image_size: int,
) -> None:
	detections: pd.DataFrame = analysis["detections"]
	total = len(detections)
	class_names = detections["Predicted class"].astype(str) if total else pd.Series(dtype=str)
	fresh_count = int(class_names.isin(FRESH_CLASSES).sum())
	rotten_count = int(class_names.isin(ROTTEN_CLASSES).sum())
	average_confidence = (
		float(detections["Confidence"].mean()) if total else None
	)

	st.subheader(model_label)
	st.caption(f"Checkpoint: `{model_filename}`")
	annotated = make_annotated_image(
		analysis["prediction"], show_masks and supports_masks, show_boxes
	)
	metric_columns = st.columns(4)
	metric_columns[0].metric("Detected fruits", total)
	metric_columns[1].metric("Fresh-class predictions", fresh_count)
	metric_columns[2].metric("Rotten-class predictions", rotten_count)
	metric_columns[3].metric(
		"Average confidence",
		f"{average_confidence:.1%}" if average_confidence is not None else "N/A",
	)

	original_column, result_column = st.columns(2)
	with original_column:
		st.image(original_image, caption="Original image", use_container_width=True)
	with result_column:
		st.image(
			annotated,
			caption=(
				"Whole-fruit segmentation predictions"
				if supports_masks
				else "Detection predictions (this model does not produce masks)"
			),
			use_container_width=True,
		)

	if analysis["inference_ms"] is not None:
		st.caption(f"Model inference: {analysis['inference_ms']:.1f} ms")
	if (
		analysis["confidence_threshold"] != confidence_threshold
		or analysis["image_size"] != image_size
	):
		st.caption("Settings changed. Select Analyze image to run inference with the current settings.")

	if total:
		st.dataframe(detections, hide_index=True, use_container_width=True)
	else:
		st.info("No fruits were detected at the selected confidence threshold.")

	csv_bytes = detections.to_csv(index=False).encode("utf-8")
	png_buffer = io.BytesIO()
	annotated.save(png_buffer, format="PNG")
	download_columns = st.columns(2)
	download_columns[0].download_button(
		"Download detection table (CSV)",
		data=csv_bytes,
		file_name=f"fruit_detections_{model_filename.removesuffix('.pt')}.csv",
		mime="text/csv",
		use_container_width=True,
	)
	download_columns[1].download_button(
		"Download annotated image (PNG)",
		data=png_buffer.getvalue(),
		file_name=f"fruit_quality_result_{model_filename.removesuffix('.pt')}.png",
		mime="image/png",
		use_container_width=True,
	)

	st.caption(
		"Fresh/rotten labels are model predictions, not verified judgments of food safety. "
		"The confidence score describes the model's detection confidence, not the amount of rot."
	)


st.markdown(
	"""
	<style>
	:root { color-scheme: dark; }
	.stApp { background: #101512; }
	[data-testid="stSidebar"] { background: #171e19; }
	.block-container { padding-top: 2.1rem; padding-bottom: 3rem; }
	h1 { color: #f0f4ef; letter-spacing: 0; }
	h2, h3 { color: #d8e4da; letter-spacing: 0; }
	[data-testid="stMetric"] { background: #1a221d; border: 1px solid #35443a;
		border-radius: 6px; padding: 0.85rem 1rem; }
	[data-testid="stAlert"] { border-radius: 6px; }
	</style>
	""",
	unsafe_allow_html=True,
)

st.title("Fruit Quality Analyzer")
st.write(
	"Run one fruit image through four trained models and compare their independent predictions."
)

with st.sidebar:
	st.header("Analysis settings")
	confidence_threshold = st.slider(
		"Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05
	)
	image_size = st.select_slider(
		"Inference image size",
		options=[320, 480, 640, 768, 960, 1280],
		value=640,
	)
	show_masks = st.checkbox("Show whole-fruit masks", value=True)
	show_boxes = st.checkbox("Show bounding boxes and labels", value=True)
	st.divider()
	st.caption(
		"Three models segment whole fruits; the detection model does not produce masks. "
		"Rotten-region percentages are unavailable because none of these models segments "
		"individual rotten patches."
	)

uploaded_file = st.file_uploader(
	"Choose a fruit image",
	type=["jpg", "jpeg", "png", "webp"],
	accept_multiple_files=False,
)

if uploaded_file is None:
	st.info("Upload a JPG, JPEG, PNG, or WEBP image to begin.")
else:
	upload_bytes = uploaded_file.getvalue()
	upload_hash = hashlib.sha256(upload_bytes).hexdigest()
	try:
		original_image = read_uploaded_image(upload_bytes)
	except (UnidentifiedImageError, OSError, ValueError) as exc:
		LOGGER.exception("Unable to read uploaded image %s", uploaded_file.name)
		st.error("This file could not be read as a valid image. Try another JPG, PNG, or WEBP file.")
	else:
		st.image(original_image, caption="Original image", use_container_width=True)
		run_inference = st.button(
			"Analyze with all four models", type="primary", use_container_width=True
		)
		if run_inference:
			try:
				models = load_models()
				analyses: dict[str, Any] = {}
				with st.spinner("Running inference with all four models..."):
					for config in MODEL_CONFIGS:
						prediction = models[config["key"]].predict(
							source=original_image,
							conf=confidence_threshold,
							imgsz=image_size,
							verbose=False,
						)[0]
						detections = build_detection_table(prediction)
						speed = getattr(prediction, "speed", None)
						inference_ms = (
							float(speed["inference"])
							if isinstance(speed, dict)
							and speed.get("inference") is not None
							else None
						)
						analyses[config["key"]] = {
							"detections": detections,
							"prediction": prediction,
							"inference_ms": inference_ms,
						}
				st.session_state["fruit_analysis"] = {
					"upload_hash": upload_hash,
					"analyses": analyses,
					"confidence_threshold": confidence_threshold,
					"image_size": image_size,
				}
			except Exception as exc:
				LOGGER.exception("Fruit image inference failed for one or more models")
				st.session_state.pop("fruit_analysis", None)
				st.error(f"Analysis could not be completed: {exc}")

		analysis = st.session_state.get("fruit_analysis")
		if analysis and analysis.get("upload_hash") == upload_hash:
			st.subheader("Analysis results by model")
			result_tabs = st.tabs([config["label"] for config in MODEL_CONFIGS])
			for config, result_tab in zip(MODEL_CONFIGS, result_tabs):
				with result_tab:
					render_results(
						config["label"],
						config["filename"],
						analysis["analyses"][config["key"]],
						original_image,
						show_masks,
						show_boxes,
						config["task"] == "segment",
						confidence_threshold,
						image_size,
					)

st.divider()
st.caption(
	"Each model reports its own fruit detections. Segmentation masks cover whole fruits, "
	"not rotten regions; predictions are not verified judgments of food safety."
)
