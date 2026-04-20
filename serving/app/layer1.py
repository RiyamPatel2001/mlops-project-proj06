"""
Layer 1 multi-model registry and demand router.

All three serving tiers are loaded during startup and kept warm:

* ``good``  -> MiniLM
* ``fast``  -> FastText
* ``cheap`` -> TF-IDF + Logistic Regression

Incoming request demand is measured over a rolling time window. The router keeps
``good`` as the default tier at low load, promotes to ``fast`` under medium
load, and promotes to ``cheap`` under high load. A switch is only committed once
the target model reports itself ready; requests already assigned to another tier
continue on that tier until they finish.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from app.config import (
    LABEL_CLASSES,
    LAYER1_HF_MAX_LENGTH,
    MLFLOW_TRACKING_URI,
    ROUTER_DEMAND_WINDOW_SECONDS,
    ROUTER_HIGH_DEMAND_RPS,
    ROUTER_MEDIUM_DEMAND_RPS,
    ROUTER_SWITCH_COOLDOWN_SECONDS,
    TIER_CHEAP_ARTIFACT_PATH,
    TIER_CHEAP_MODEL_KIND,
    TIER_CHEAP_MODEL_NAME,
    TIER_CHEAP_RUN_ID,
    TIER_FAST_ARTIFACT_PATH,
    TIER_FAST_MODEL_KIND,
    TIER_FAST_MODEL_NAME,
    TIER_FAST_RUN_ID,
    TIER_GOOD_ARTIFACT_PATH,
    TIER_GOOD_MODEL_KIND,
    TIER_GOOD_MODEL_NAME,
    TIER_GOOD_RUN_ID,
)

logger = logging.getLogger(__name__)


class Tier(str, Enum):
    GOOD = "good"
    FAST = "fast"
    CHEAP = "cheap"


class DemandLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class TierConfig:
    tier: Tier
    name: str
    kind: str
    run_id: str
    artifact_path: str


@dataclass
class ModelRuntime:
    config: TierConfig
    predictor: Callable[[str], tuple[str, float]]
    version: str
    ready: bool = False
    active_requests: int = 0
    load_error: Optional[str] = None


@dataclass(frozen=True)
class PredictionResult:
    category: str
    confidence: float
    tier: str
    demand_level: str
    model_name: str
    model_kind: str
    model_version: str


@dataclass(frozen=True)
class ModelStatus:
    tier: str
    model_name: str
    model_kind: str
    model_version: str
    ready: bool
    active_requests: int
    load_error: Optional[str]


@dataclass(frozen=True)
class RouterSnapshot:
    active_tier: str
    active_model_name: str
    active_model_version: str
    pending_tier: Optional[str]
    demand_level: str
    request_rate_rps: float
    models: list[ModelStatus] = field(default_factory=list)


class MultiModelRegistry:
    def __init__(
        self,
        configs: list[TierConfig],
        demand_window_seconds: int,
        medium_demand_rps: float,
        high_demand_rps: float,
        switch_cooldown_seconds: float,
    ) -> None:
        self._configs = {cfg.tier: cfg for cfg in configs}
        self._demand_window_seconds = max(1, demand_window_seconds)
        self._medium_demand_rps = medium_demand_rps
        self._high_demand_rps = max(high_demand_rps, medium_demand_rps)
        self._switch_cooldown_seconds = max(0.0, switch_cooldown_seconds)

        self._models: dict[Tier, ModelRuntime] = {}
        self._active_tier: Tier = Tier.GOOD
        self._pending_tier: Optional[Tier] = None
        self._request_times: deque[float] = deque()
        self._last_switch_at: float = 0.0
        self._lock = threading.RLock()

    def load_all(self) -> None:
        loaded: dict[Tier, ModelRuntime] = {}
        errors: list[str] = []

        for tier in (Tier.GOOD, Tier.FAST, Tier.CHEAP):
            cfg = self._configs[tier]
            try:
                runtime = self._load_runtime(cfg)
                loaded[tier] = runtime
                logger.info(
                    "Loaded tier=%s model=%s kind=%s version=%s",
                    cfg.tier.value,
                    cfg.name,
                    cfg.kind,
                    runtime.version,
                )
            except Exception as exc:
                message = f"{cfg.tier.value}:{cfg.name} failed to load: {exc}"
                logger.exception("Layer 1 tier load failed: %s", message)
                errors.append(message)

        with self._lock:
            self._models = loaded
            self._active_tier = Tier.GOOD
            self._pending_tier = None
            self._request_times.clear()
            self._last_switch_at = time.monotonic()

        if errors:
            raise RuntimeError(
                "Multi-model startup failed; all tiers must be ready. "
                + " | ".join(errors)
            )

        missing = [tier.value for tier in (Tier.GOOD, Tier.FAST, Tier.CHEAP) if tier not in loaded]
        if missing:
            raise RuntimeError(f"Missing loaded tiers: {', '.join(missing)}")

    def install_runtime_for_tests(self, tier: Tier, runtime: ModelRuntime) -> None:
        with self._lock:
            self._models[tier] = runtime
            if tier == Tier.GOOD and self._active_tier not in self._models:
                self._active_tier = Tier.GOOD

    def predict(self, text: str) -> PredictionResult:
        now = time.monotonic()

        with self._lock:
            self._register_request_locked(now)
            self._maybe_switch_locked(now)
            runtime = self._models[self._active_tier]
            if not runtime.ready:
                runtime = self._best_ready_runtime_locked()
            runtime.active_requests += 1
            demand_level = self._current_demand_level_locked()

        try:
            category, confidence = runtime.predictor(text)
        finally:
            with self._lock:
                runtime.active_requests = max(0, runtime.active_requests - 1)

        return PredictionResult(
            category=category,
            confidence=round(confidence, 4),
            tier=runtime.config.tier.value,
            demand_level=demand_level.value,
            model_name=runtime.config.name,
            model_kind=runtime.config.kind,
            model_version=runtime.version,
        )

    def get_model_version(self) -> str:
        snapshot = self.get_router_snapshot()
        return snapshot.active_model_version

    def get_router_snapshot(self) -> RouterSnapshot:
        with self._lock:
            self._trim_request_times_locked(time.monotonic())

            active_runtime = self._models[self._active_tier]
            models = [
                ModelStatus(
                    tier=tier.value,
                    model_name=runtime.config.name,
                    model_kind=runtime.config.kind,
                    model_version=runtime.version,
                    ready=runtime.ready,
                    active_requests=runtime.active_requests,
                    load_error=runtime.load_error,
                )
                for tier, runtime in self._models.items()
            ]

            return RouterSnapshot(
                active_tier=self._active_tier.value,
                active_model_name=active_runtime.config.name,
                active_model_version=active_runtime.version,
                pending_tier=self._pending_tier.value if self._pending_tier else None,
                demand_level=self._current_demand_level_locked().value,
                request_rate_rps=round(self._current_rps_locked(), 4),
                models=models,
            )

    def _load_runtime(self, cfg: TierConfig) -> ModelRuntime:
        if cfg.kind == "hf":
            predictor, version = _load_hf_runtime(cfg)
        elif cfg.kind == "fasttext":
            predictor, version = _load_fasttext_runtime(cfg)
        elif cfg.kind == "sklearn":
            predictor, version = _load_sklearn_runtime(cfg)
        else:
            raise ValueError(f"Unsupported model kind for tier {cfg.tier.value}: {cfg.kind}")

        runtime = ModelRuntime(
            config=cfg,
            predictor=predictor,
            version=version,
            ready=False,
        )

        # Warm-up is the readiness signal used by the router before any traffic
        # can move onto the tier.
        predictor("WARMUP")
        runtime.ready = True
        return runtime

    def _register_request_locked(self, now: float) -> None:
        self._request_times.append(now)
        self._trim_request_times_locked(now)

    def _trim_request_times_locked(self, now: float) -> None:
        cutoff = now - self._demand_window_seconds
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()

    def _current_rps_locked(self) -> float:
        return len(self._request_times) / float(self._demand_window_seconds)

    def _current_demand_level_locked(self) -> DemandLevel:
        current_rps = self._current_rps_locked()
        if current_rps >= self._high_demand_rps:
            return DemandLevel.HIGH
        if current_rps >= self._medium_demand_rps:
            return DemandLevel.MEDIUM
        return DemandLevel.LOW

    def _target_tier_for_demand_locked(self) -> Tier:
        level = self._current_demand_level_locked()
        if level == DemandLevel.HIGH:
            return Tier.CHEAP
        if level == DemandLevel.MEDIUM:
            return Tier.FAST
        return Tier.GOOD

    def _maybe_switch_locked(self, now: float) -> None:
        target_tier = self._target_tier_for_demand_locked()
        if target_tier == self._active_tier:
            self._pending_tier = None
            return

        runtime = self._models.get(target_tier)
        if runtime is None or not runtime.ready:
            self._pending_tier = target_tier
            return

        if now - self._last_switch_at < self._switch_cooldown_seconds:
            self._pending_tier = target_tier
            return

        logger.info(
            "Demand tier switch: %s -> %s (rps=%.3f)",
            self._active_tier.value,
            target_tier.value,
            self._current_rps_locked(),
        )
        self._active_tier = target_tier
        self._pending_tier = None
        self._last_switch_at = now

    def _best_ready_runtime_locked(self) -> ModelRuntime:
        preferred = [self._active_tier, Tier.GOOD, Tier.FAST, Tier.CHEAP]
        for tier in preferred:
            runtime = self._models.get(tier)
            if runtime and runtime.ready:
                return runtime
        raise RuntimeError("No ready Layer 1 model is available")


def _download_artifact(run_id: str, artifact_path: str) -> str:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
    )


def _load_hf_runtime(cfg: TierConfig) -> tuple[Callable[[str], tuple[str, float]], str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_dir = _download_artifact(cfg.run_id, cfg.artifact_path)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(local_dir)
    model.eval()

    cfg_id2label = getattr(model.config, "id2label", None)
    if cfg_id2label and not all(str(v).startswith("LABEL_") for v in cfg_id2label.values()):
        id2label = [cfg_id2label[i] for i in sorted(cfg_id2label, key=lambda key: int(key))]
    else:
        id2label = list(LABEL_CLASSES)

    def predict(text: str) -> tuple[str, float]:
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=LAYER1_HF_MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item())
        label = id2label[top_idx] if top_idx < len(id2label) else "Other"
        if label not in LABEL_CLASSES:
            label = "Other"
        return label, confidence

    version = f"mlflow-run:{cfg.run_id}/{cfg.artifact_path}"
    return predict, version


def _load_fasttext_runtime(cfg: TierConfig) -> tuple[Callable[[str], tuple[str, float]], str]:
    import fasttext

    local_path = _download_artifact(cfg.run_id, cfg.artifact_path)
    model_path = _resolve_fasttext_path(local_path)
    ft_model = fasttext.load_model(model_path)
    label_lookup = {
        f"__label__{label.replace(' ', '_')}": label for label in LABEL_CLASSES
    }

    def predict(text: str) -> tuple[str, float]:
        labels, probs = ft_model.predict(text.replace("\n", " "), k=1)
        raw_label = labels[0]
        label = label_lookup.get(raw_label, raw_label.replace("__label__", "").replace("_", " "))
        if label not in LABEL_CLASSES:
            label = "Other"
        return label, float(probs[0])

    version = f"mlflow-run:{cfg.run_id}/{Path(model_path).name}"
    return predict, version


def _load_sklearn_runtime(cfg: TierConfig) -> tuple[Callable[[str], tuple[str, float]], str]:
    import joblib
    import numpy as np

    local_path = _download_artifact(cfg.run_id, cfg.artifact_path)
    payload = joblib.load(local_path)

    vectorizer = None
    classifier = payload
    if isinstance(payload, dict):
        vectorizer = payload.get("vectorizer")
        classifier = payload.get("classifier", payload)

    def predict(text: str) -> tuple[str, float]:
        if vectorizer is not None:
            features = vectorizer.transform([text])
            probs = classifier.predict_proba(features)[0]
        else:
            probs = classifier.predict_proba([text])[0]

        top_idx = int(np.argmax(probs))
        classes = getattr(classifier, "classes_", None)
        pred_label = classes[top_idx] if classes is not None else top_idx
        label = _resolve_label(pred_label)
        confidence = float(probs[top_idx])
        return label, confidence

    version = f"mlflow-run:{cfg.run_id}/{Path(local_path).name}"
    return predict, version


def _resolve_fasttext_path(local_path: str) -> str:
    path = Path(local_path)
    if path.is_file():
        return str(path)
    for suffix in (".bin", ".ftz"):
        matches = sorted(path.rglob(f"*{suffix}"))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"Could not find a FastText model beneath {local_path}")


def _resolve_label(pred_label: object) -> str:
    if isinstance(pred_label, str) and pred_label in LABEL_CLASSES:
        return pred_label

    if isinstance(pred_label, (int,)) and 0 <= pred_label < len(LABEL_CLASSES):
        return LABEL_CLASSES[pred_label]

    try:
        index = int(pred_label)
    except (TypeError, ValueError):
        return "Other"

    if 0 <= index < len(LABEL_CLASSES):
        return LABEL_CLASSES[index]
    return "Other"


_registry = MultiModelRegistry(
    configs=[
        TierConfig(
            tier=Tier.GOOD,
            name=TIER_GOOD_MODEL_NAME,
            kind=TIER_GOOD_MODEL_KIND,
            run_id=TIER_GOOD_RUN_ID,
            artifact_path=TIER_GOOD_ARTIFACT_PATH,
        ),
        TierConfig(
            tier=Tier.FAST,
            name=TIER_FAST_MODEL_NAME,
            kind=TIER_FAST_MODEL_KIND,
            run_id=TIER_FAST_RUN_ID,
            artifact_path=TIER_FAST_ARTIFACT_PATH,
        ),
        TierConfig(
            tier=Tier.CHEAP,
            name=TIER_CHEAP_MODEL_NAME,
            kind=TIER_CHEAP_MODEL_KIND,
            run_id=TIER_CHEAP_RUN_ID,
            artifact_path=TIER_CHEAP_ARTIFACT_PATH,
        ),
    ],
    demand_window_seconds=ROUTER_DEMAND_WINDOW_SECONDS,
    medium_demand_rps=ROUTER_MEDIUM_DEMAND_RPS,
    high_demand_rps=ROUTER_HIGH_DEMAND_RPS,
    switch_cooldown_seconds=ROUTER_SWITCH_COOLDOWN_SECONDS,
)


def load_model() -> None:
    _registry.load_all()


def predict(text: str) -> PredictionResult:
    return _registry.predict(text)


def get_model_version() -> str:
    return _registry.get_model_version()


def get_router_snapshot() -> RouterSnapshot:
    return _registry.get_router_snapshot()
