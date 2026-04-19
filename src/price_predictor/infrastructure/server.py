"""FastAPI prediction service for card price estimation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from price_predictor.application.transformer_inference import (
    predict_shifted_log,
    shifted_log_to_eur,
)
from price_predictor.domain.card_name_resolver import CardNameResolver
from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.domain.entities import Card
from price_predictor.domain.tokenizer import MtgTokenizer, extract_card_name
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.converted_card_parser import parse_converted_text

logger = logging.getLogger(__name__)


def _build_log_entry(
    status_code: int,
    latency_ms: float,
    **extra: Any,
) -> dict[str, Any]:
    """Build a structured log entry for an evaluate request."""
    entry: dict[str, Any] = {
        "event": "evaluate_request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
    }
    entry.update(extra)
    return entry


def _resolve_printing_data(
    body: str, resolver: CardNameResolver | None
) -> PrintingData | None:
    """Look up PrintingData for the card in the request, or None when unknown.

    Returning None lets feature engineering use its zeroed branch instead of
    the (real-looking) PrintingData.defaults() values.
    """
    if resolver is None:
        return None
    card_name = extract_card_name(body)
    if not card_name:
        return None
    if resolver.canonicalize(card_name) is None:
        return None
    return resolver.lookup_printing_data(card_name)


def _run_sklearn_prediction(
    artifact: dict[str, Any], card: Card
) -> tuple[float, str]:
    """Run sklearn inference and return (predicted_price_eur, model_version)."""
    fe = artifact["feature_engineering"]
    model = artifact["model"]
    X = fe.transform([card])
    log_price = model.predict(X)[0]
    return round(float(np.exp(log_price)), 2), artifact["model_version"]


def _run_transformer_prediction(
    artifact: dict[str, Any] | None,
    body: str,
    printing_data: PrintingData | None,
    tokenizer: MtgTokenizer | None,
) -> dict[str, Any] | None:
    """Run optional transformer inference; return None on absence or failure."""
    if artifact is None:
        return None
    try:
        if tokenizer is None:
            raise RuntimeError("Tokenizer not loaded — run 'vocabulary' first")
        config = artifact["config"]
        shifted_log_pred = predict_shifted_log(
            artifact["model"], tokenizer, body, printing_data, config,
        )
        price = round(shifted_log_to_eur(shifted_log_pred, config.log_offset), 2)
        return {
            "predicted_price_eur": price,
            "model_version": artifact["model_version"],
        }
    except Exception as e:
        logger.warning("Transformer prediction failed: %s", e)
        return None


def _build_response(
    sklearn_price: float,
    sklearn_version: str,
    transformer: dict[str, Any] | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "sklearn": {
                "predicted_price_eur": sklearn_price,
                "model_version": sklearn_version,
            },
            "transformer": transformer,
        },
    )


def create_app(
    model_artifact: dict[str, Any],
    transformer_artifact: dict[str, Any] | None = None,
    metadata_map: dict | None = None,
    tokenizer: Any | None = None,
) -> FastAPI:
    """Create a FastAPI application with the given model artifact(s).

    Args:
        model_artifact: Dict with 'model', 'feature_engineering', and 'model_version' keys.
        transformer_artifact: Optional dict with 'model', 'config', and 'model_version' keys.
        tokenizer: Optional MtgTokenizer for transformer predictions.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="Price Predictor Service")
    app.state.model_artifact = model_artifact
    app.state.transformer_artifact = transformer_artifact
    app.state.metadata_map = metadata_map or {}
    app.state.card_name_resolver = (
        CardNameResolver(metadata_map=metadata_map) if metadata_map else None
    )
    app.state.tokenizer = tokenizer

    @app.post("/api/v1/predict")
    async def predict(request: Request) -> Response:
        start = time.perf_counter()
        body = (await request.body()).decode("utf-8")

        printing_data = _resolve_printing_data(
            body, request.app.state.card_name_resolver,
        )

        converted = ConvertedCardText(body)

        try:
            card = parse_converted_text(converted)
            if printing_data is not None:
                card = replace(card, printing_data=printing_data)
        except (ValueError, TypeError) as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(json.dumps(_build_log_entry(
                status_code=400,
                latency_ms=latency_ms,
                error=str(e),
            )))
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to parse converted card text: {e}"},
            )

        try:
            sklearn_price, sklearn_version = _run_sklearn_prediction(
                request.app.state.model_artifact, card,
            )
            transformer_result = _run_transformer_prediction(
                request.app.state.transformer_artifact,
                body,
                printing_data,
                request.app.state.tokenizer,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            log_extra: dict[str, Any] = {
                "card_name": card.name,
                "card_types": list(card.types),
                "card_mana_cost": converted.mana_cost_line(),
                "sklearn_predicted_price_eur": sklearn_price,
                "sklearn_model_version": sklearn_version,
            }
            if transformer_result:
                log_extra["transformer_predicted_price_eur"] = (
                    transformer_result["predicted_price_eur"]
                )
                log_extra["transformer_model_version"] = (
                    transformer_result["model_version"]
                )
            logger.info(json.dumps(_build_log_entry(
                status_code=200,
                latency_ms=latency_ms,
                **log_extra,
            )))

            return _build_response(sklearn_price, sklearn_version, transformer_result)
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(json.dumps(_build_log_entry(
                status_code=500,
                latency_ms=latency_ms,
                error=str(e),
            )))
            logger.exception("Prediction failed")
            return JSONResponse(
                status_code=500,
                content={"error": f"Prediction failed: {e}"},
            )

    return app
