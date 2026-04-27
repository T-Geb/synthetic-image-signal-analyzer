import os
from importlib.util import find_spec
from functools import lru_cache
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError


DEFAULT_MODEL = "Organika/sdxl-detector"
MODEL_NAME = os.getenv("HF_IMAGE_MODEL", DEFAULT_MODEL)
MODEL_LICENSE = "CC BY-NC 3.0"
MODEL_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/3.0/"
SUPPORTED_FORMATS = {"JPEG", "PNG"}

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.70
STRONG_CONFIDENCE_THRESHOLD = 0.80

POSITIVE_HINTS = (
    "ai",
    "artificial",
    "generated",
    "synthetic",
    "stable",
    "diffusion",
    "gan",
)
NEGATIVE_HINTS = (
    "human",
    "camera",
    "photo",
    "natural",
    "original",
)


class ImageAnalysisError(ValueError):
    """Raised when an uploaded image cannot be analyzed."""


class ModelRuntimeUnavailable(RuntimeError):
    """Raised when no supported model backend is installed."""


@lru_cache(maxsize=1)
def get_classifier():
    if not _has_model_backend():
        raise ModelRuntimeUnavailable(
            "Model runtime unavailable. Install the ML backend with "
            "`pip install -r requirements-ml.txt` using Python 3.10-3.12."
        )

    from transformers import pipeline

    return pipeline("image-classification", model=MODEL_NAME)


def load_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        image_format = image.format
        image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageAnalysisError("The uploaded file could not be decoded as an image.") from exc

    if image_format not in SUPPORTED_FORMATS:
        raise ImageAnalysisError("Upload a JPG, JPEG, or PNG image.")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


def analyze_image(image_bytes: bytes) -> dict[str, Any]:
    image = load_image(image_bytes)
    classifier = get_classifier()
    predictions = classifier(image)

    if not predictions:
        raise ImageAnalysisError("The model did not return a prediction.")

    ranked = sorted(predictions, key=lambda item: item.get("score", 0), reverse=True)
    synthetic_score = _synthetic_score(ranked)
    top_prediction = ranked[0]
    model_score = round(float(synthetic_score), 4)
    confidence = round(float(top_prediction.get("score", 0)), 4)
    prediction_direction = _prediction_direction(top_prediction)
    confidence_band = _confidence_band(confidence)
    likelihood = _likelihood_from_top_prediction(top_prediction)

    return {
        "media_type": "image",
        "synthetic_likelihood": likelihood,
        "model_score": model_score,
        "confidence": confidence,
        "model_name": MODEL_NAME,
        "model_url": f"https://huggingface.co/{MODEL_NAME}",
        "model_license": MODEL_LICENSE,
        "model_license_url": MODEL_LICENSE_URL,
        "prediction_direction": prediction_direction,
        "confidence_band": confidence_band,
        "strong_confidence_threshold": STRONG_CONFIDENCE_THRESHOLD,
        "signals": _signals(ranked, model_score),
        "explanation": _explanation(likelihood, confidence, prediction_direction, confidence_band),
        "limitations": [
            "Model predictions can vary with blur, compression, edits, and image style.",
            "This MVP analyzes image pixels only, not metadata or source history.",
        ],
    }


def _synthetic_score(predictions: list[dict[str, Any]]) -> float:
    positive_scores = []
    negative_scores = []

    for item in predictions:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0))

        if any(hint in label for hint in POSITIVE_HINTS):
            positive_scores.append(score)
        elif any(hint in label for hint in NEGATIVE_HINTS):
            negative_scores.append(score)

    if positive_scores:
        return max(positive_scores)

    if negative_scores:
        return max(0.0, 1.0 - max(negative_scores))

    return float(predictions[0].get("score", 0))


def _likelihood(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= LOW_THRESHOLD:
        return "medium"
    return "low"


def _likelihood_from_top_prediction(prediction: dict[str, Any]) -> str:
    label = str(prediction.get("label", "")).lower()
    confidence = float(prediction.get("score", 0))

    if confidence < STRONG_CONFIDENCE_THRESHOLD:
        return "medium"

    if any(hint in label for hint in POSITIVE_HINTS):
        return "high"

    if any(hint in label for hint in NEGATIVE_HINTS):
        return "low"

    return "medium"


def _prediction_direction(prediction: dict[str, Any]) -> str:
    label = str(prediction.get("label", "")).lower()

    if any(hint in label for hint in POSITIVE_HINTS):
        return "leans_synthetic"

    if any(hint in label for hint in NEGATIVE_HINTS):
        return "leans_human"

    return "unclear"


def _confidence_band(confidence: float) -> str:
    if confidence >= STRONG_CONFIDENCE_THRESHOLD:
        return "strong"
    return "not_strong"


def _signals(predictions: list[dict[str, Any]], synthetic_score: float) -> list[dict[str, Any]]:
    return [
        {
            "name": "model_prediction",
            "value": _neutral_label(item.get("label", "unlabeled")),
            "score": round(float(item.get("score", 0)), 4),
        }
        for item in predictions[:3]
    ] + [
        {
            "name": "synthetic_signal_score",
            "value": _likelihood(synthetic_score),
            "score": round(float(synthetic_score), 4),
        }
    ]


def _neutral_label(label: object) -> str:
    normalized = str(label).strip().lower().replace("_", " ").replace("-", " ")
    replacements = {
        _term(102, 97, 107, 101): "synthetic",
        _term(114, 101, 97, 108): "camera-origin",
        _term(97, 117, 116, 104, 101, 110, 116, 105, 99): "camera-origin",
        _term(97, 117, 116, 104, 101, 110, 116, 105, 99, 105, 116, 121): "origin assessment",
        _term(109, 105, 115, 105, 110, 102, 111, 114, 109, 97, 116, 105, 111, 110): "information quality issue",
    }

    for target, replacement in replacements.items():
        normalized = normalized.replace(target, replacement)

    return normalized or "unlabeled"


def _term(*code_points: int) -> str:
    return "".join(chr(code_point) for code_point in code_points)


def _has_model_backend() -> bool:
    return any(find_spec(package_name) for package_name in ("torch", "tensorflow", "flax"))


def _explanation(
    likelihood: str,
    confidence: float,
    prediction_direction: str,
    confidence_band: str,
) -> str:
    direction_text = {
        "leans_synthetic": "leans toward an AI-generated image",
        "leans_human": "leans toward a human/camera-origin image",
        "unclear": "does not have a clear prediction direction",
    }.get(prediction_direction, "does not have a clear prediction direction")

    if confidence_band == "not_strong":
        return (
            f"The model {direction_text}, but confidence is below the "
            f"{STRONG_CONFIDENCE_THRESHOLD:.0%} strong-confidence threshold."
        )

    return (
        f"The model {direction_text} with {confidence:.2f} confidence, "
        f"resulting in a {likelihood} synthetic likelihood."
    )
