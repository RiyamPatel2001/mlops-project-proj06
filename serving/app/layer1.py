"""
Layer 1 multi-model registry with request-aware routing.

All three serving tiers are loaded during startup and kept warm:

* ``good``  -> MiniLM for interactive, user-facing predictions
* ``fast``  -> FastText for normal bulk imports
* ``cheap`` -> TF-IDF + Logistic Regression as overload fallback

Routing is no longer a single global switch for the whole server. Instead:

* interactive requests always prefer ``good``
* bulk requests default to ``fast``
* new bulk batches move to ``cheap`` only after sustained service overload
* a bulk ``batch_id`` is pinned to one tier for a configurable TTL so one
  import job stays consistent even while other users are active
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
    ROUTER_BATCH_STICKY_TTL_SECONDS,
    ROUTER_OVERLOAD_INFLIGHT_THRESHOLD,
    ROUTER_OVERLOAD_SUSTAIN_SECONDS,
    ROUTER_REQUEST_WINDOW_SECONDS,
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


class RequestMode(str, Enum):
    INTERACTIVE = "interactive"
    BULK = "bulk"


class OverloadState(str, Enum):
    NORMAL = "normal"
    WARMING = "warming"
    ACTIVE = "active"


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


@dataclass
class BatchAssignment:
    tier: Tier
    expires_at: float


@dataclass(frozen=True)
class PredictionResult:
    category: str
    confidence: float
    tier: str
    model_name: str
    model_kind: str
    model_version: str
    request_mode: str
    batch_id: Optional[str]
    overload_state: str


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
    interactive_tier: str
    interactive_model_name: str
    bulk_tier: str
    bulk_model_name: str
    pending_tier: Optional[str]
    demand_level: str
    overload_state: str
    active_batch_count: int
    total_inflight_requests: int
    last_request_mode: str
    request_rate_rps: float
    models: list[ModelStatus] = field(default_factory=list)


class MultiModelRegistry:
    def __init__(
        self,
        configs: list[TierConfig],
        request_window_seconds: int,
        batch_sticky_ttl_seconds: int,
        overload_inflight_threshold: int,
        overload_sustain_seconds: float,
    ) -> None:
        self._configs = {cfg.tier: cfg for cfg in configs}
        self._request_window_seconds = max(1, request_window_seconds)
        self._batch_sticky_ttl_seconds = max(1, batch_sticky_ttl_seconds)
        self._overload_inflight_threshold = max(1, overload_inflight_threshold)
        self._overload_sustain_seconds = max(0.0, overload_sustain_seconds)

        self._models: dict[Tier, ModelRuntime] = {}
        self._request_times: deque[float] = deque()
        self._batch_assignments: dict[str, BatchAssignment] = {}
        self._total_inflight_requests: int = 0
        self._overload_started_at: Optional[float] = None
        self._last_routed_tier: Tier = Tier.GOOD
        self._last_request_mode: RequestMode = RequestMode.INTERACTIVE
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
            self._request_times.clear()
            self._batch_assignments.clear()
            self._total_inflight_requests = 0
            self._overload_started_at = None
            self._last_routed_tier = Tier.GOOD
            self._last_request_mode = RequestMode.INTERACTIVE

        if errors:
            raise RuntimeError(
                "Multi-model startup failed; all tiers must be ready. "
                + " | ".join(errors)
            )

        missing = [
            tier.value
            for tier in (Tier.GOOD, Tier.FAST, Tier.CHEAP)
            if tier not in loaded
        ]
        if missing:
            raise RuntimeError(f"Missing loaded tiers: {', '.join(missing)}")

    def install_runtime_for_tests(self, tier: Tier, runtime: ModelRuntime) -> None:
        with self._lock:
            self._models[tier] = runtime

    def predict(
        self,
        text: str,
        request_mode: str = RequestMode.INTERACTIVE.value,
        batch_id: Optional[str] = None,
    ) -> PredictionResult:
        now = time.monotonic()
        mode = _normalize_request_mode(request_mode)

        with self._lock:
            self._cleanup_batch_assignments_locked(now)
            self._register_request_locked(now)
            selected_tier = self._select_tier_locked(mode, batch_id, now)
            runtime = self._preferred_runtime_locked(selected_tier)
            runtime.active_requests += 1
            self._total_inflight_requests += 1
            overload_state = self._current_overload_state_locked(now)
            self._last_routed_tier = runtime.config.tier
            self._last_request_mode = mode

        try:
            category, confidence = runtime.predictor(text)
        finally:
            with self._lock:
                runtime.active_requests = max(0, runtime.active_requests - 1)
                self._total_inflight_requests = max(
                    0,
                    self._total_inflight_requests - 1,
                )
                self._current_overload_state_locked(time.monotonic())

        return PredictionResult(
            category=category,
            confidence=round(confidence, 4),
            tier=runtime.config.tier.value,
            model_name=runtime.config.name,
            model_kind=runtime.config.kind,
            model_version=runtime.version,
            request_mode=mode.value,
            batch_id=batch_id,
            overload_state=overload_state.value,
        )

    def get_model_version(self) -> str:
        return self.get_router_snapshot().active_model_version

    def get_router_snapshot(self) -> RouterSnapshot:
        with self._lock:
            now = time.monotonic()
            self._trim_request_times_locked(now)
            self._cleanup_batch_assignments_locked(now)
            overload_state = self._current_overload_state_locked(now)

            interactive_runtime = self._preferred_runtime_locked(Tier.GOOD)
            bulk_runtime = self._preferred_runtime_locked(self._bulk_tier_locked(now))
            last_runtime = self._preferred_runtime_locked(self._last_routed_tier)

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
                for tier in (Tier.GOOD, Tier.FAST, Tier.CHEAP)
                if (runtime := self._models.get(tier)) is not None
            ]

            return RouterSnapshot(
                active_tier=last_runtime.config.tier.value,
                active_model_name=last_runtime.config.name,
                active_model_version=last_runtime.version,
                interactive_tier=interactive_runtime.config.tier.value,
                interactive_model_name=interactive_runtime.config.name,
                bulk_tier=bulk_runtime.config.tier.value,
                bulk_model_name=bulk_runtime.config.name,
                pending_tier=(
                    Tier.CHEAP.value
                    if overload_state == OverloadState.WARMING
                    else None
                ),
                demand_level=overload_state.value,
                overload_state=overload_state.value,
                active_batch_count=len(self._batch_assignments),
                total_inflight_requests=self._total_inflight_requests,
                last_request_mode=self._last_request_mode.value,
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
            raise ValueError(
                f"Unsupported model kind for tier {cfg.tier.value}: {cfg.kind}"
            )

        runtime = ModelRuntime(
            config=cfg,
            predictor=predictor,
            version=version,
            ready=False,
        )

        predictor("WARMUP")
        runtime.ready = True
        return runtime

    def _register_request_locked(self, now: float) -> None:
        self._request_times.append(now)
        self._trim_request_times_locked(now)

    def _trim_request_times_locked(self, now: float) -> None:
        cutoff = now - self._request_window_seconds
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()

    def _cleanup_batch_assignments_locked(self, now: float) -> None:
        expired = [
            batch_id
            for batch_id, assignment in self._batch_assignments.items()
            if assignment.expires_at < now
        ]
        for batch_id in expired:
            del self._batch_assignments[batch_id]

    def _current_rps_locked(self) -> float:
        return len(self._request_times) / float(self._request_window_seconds)

    def _current_overload_state_locked(self, now: float) -> OverloadState:
        if self._total_inflight_requests >= self._overload_inflight_threshold:
            if self._overload_started_at is None:
                self._overload_started_at = now
            if now - self._overload_started_at >= self._overload_sustain_seconds:
                return OverloadState.ACTIVE
            return OverloadState.WARMING

        self._overload_started_at = None
        return OverloadState.NORMAL

    def _select_tier_locked(
        self,
        request_mode: RequestMode,
        batch_id: Optional[str],
        now: float,
    ) -> Tier:
        if request_mode == RequestMode.BULK:
            if batch_id:
                assignment = self._batch_assignments.get(batch_id)
                if assignment is not None:
                    assignment.expires_at = now + self._batch_sticky_ttl_seconds
                    return assignment.tier

            selected = self._bulk_tier_locked(now)
            if batch_id:
                self._batch_assignments[batch_id] = BatchAssignment(
                    tier=selected,
                    expires_at=now + self._batch_sticky_ttl_seconds,
                )
            return selected

        return Tier.GOOD

    def _bulk_tier_locked(self, now: float) -> Tier:
        overload_state = self._current_overload_state_locked(now)
        if overload_state == OverloadState.ACTIVE:
            return Tier.CHEAP
        return Tier.FAST

    def _preferred_runtime_locked(self, preferred_tier: Tier) -> ModelRuntime:
        for tier in (preferred_tier, Tier.GOOD, Tier.FAST, Tier.CHEAP):
            runtime = self._models.get(tier)
            if runtime and runtime.ready:
                return runtime
        raise RuntimeError("No ready Layer 1 model is available")


def _normalize_request_mode(request_mode: str | RequestMode) -> RequestMode:
    if isinstance(request_mode, RequestMode):
        return request_mode
    try:
        return RequestMode(request_mode)
    except ValueError:
        return RequestMode.INTERACTIVE


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
    if cfg_id2label and not all(
        str(v).startswith("LABEL_") for v in cfg_id2label.values()
    ):
        id2label = [
            cfg_id2label[i]
            for i in sorted(cfg_id2label, key=lambda key: int(key))
        ]
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
        label = label_lookup.get(
            raw_label,
            raw_label.replace("__label__", "").replace("_", " "),
        )
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

    if isinstance(pred_label, int) and 0 <= pred_label < len(LABEL_CLASSES):
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
    request_window_seconds=ROUTER_REQUEST_WINDOW_SECONDS,
    batch_sticky_ttl_seconds=ROUTER_BATCH_STICKY_TTL_SECONDS,
    overload_inflight_threshold=ROUTER_OVERLOAD_INFLIGHT_THRESHOLD,
    overload_sustain_seconds=ROUTER_OVERLOAD_SUSTAIN_SECONDS,
)


def load_model() -> None:
    _registry.load_all()


def predict(
    text: str,
    request_mode: str = RequestMode.INTERACTIVE.value,
    batch_id: Optional[str] = None,
) -> PredictionResult:
    return _registry.predict(text, request_mode=request_mode, batch_id=batch_id)


def get_model_version() -> str:
    return _registry.get_model_version()


def get_router_snapshot() -> RouterSnapshot:
    return _registry.get_router_snapshot()
