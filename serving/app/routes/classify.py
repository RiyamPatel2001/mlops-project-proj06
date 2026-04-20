"""POST /classify — full classification pipeline."""

import logging
import time

from fastapi import APIRouter

from app import layer1, layer2
from app.feature_computation import compute_features
from app.metrics import record_classification, record_classification_failure
from app.models import ClassifyRequest, ClassifyResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_transaction(req: ClassifyRequest) -> ClassifyResponse:
    t0 = time.perf_counter()
    try:
        features = compute_features(req.payee, req.amount, req.date)
        model_input = features["normalized_payee"]

        l1_result = layer1.predict(
            model_input,
            request_mode=req.request_mode,
            batch_id=req.batch_id,
        )

        l2_result = await layer2.classify(req.user_id, req.payee)
        if l2_result is not None:
            category, similarity = l2_result
            record_classification(
                latency=time.perf_counter() - t0,
                source="layer2",
                request_mode=req.request_mode,
                model_tier="layer2",
                model_version="layer2",
                confidence=similarity,
                category=category,
            )
            return ClassifyResponse(
                transaction_id=req.transaction_id,
                user_id=req.user_id,
                prediction_category=category,
                confidence=round(similarity, 4),
                source="layer2",
                model_version=None,
            )

        record_classification(
            latency=time.perf_counter() - t0,
            source="layer1",
            request_mode=l1_result.request_mode,
            model_tier=l1_result.tier,
            model_version=l1_result.model_version,
            confidence=l1_result.confidence,
            category=l1_result.category,
        )
        return ClassifyResponse(
            transaction_id=req.transaction_id,
            user_id=req.user_id,
            prediction_category=l1_result.category,
            confidence=l1_result.confidence,
            source="layer1",
            model_version=l1_result.model_version,
        )
    except Exception:
        logger.exception(
            "Classification failed for transaction_id=%s user_id=%s",
            req.transaction_id,
            req.user_id,
        )
        record_classification_failure(
            latency=time.perf_counter() - t0,
            request_mode=req.request_mode,
        )
        raise
