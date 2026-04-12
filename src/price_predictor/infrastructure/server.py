"""FastAPI prediction service for card price estimation."""

from __future__ import annotations

import json
import logging
import time
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
from price_predictor.domain.tokenizer import extract_card_name
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

        # Look up PrintingData from metadata_map for the card in the request.
        # When unknown, leave it None so feature engineering uses its zeroed
        # branch instead of the (real-looking) PrintingData.defaults() values.
        from price_predictor.domain.value_objects import PrintingData
        printing_data: PrintingData | None = None
        card_name_for_lookup = extract_card_name(body)
        resolver: CardNameResolver | None = request.app.state.card_name_resolver
        if card_name_for_lookup and resolver is not None:
            canonical = resolver.canonicalize(card_name_for_lookup)
            if canonical is not None:
                printing_data = resolver.lookup_printing_data(card_name_for_lookup)

        try:
            card = parse_converted_text(body)
            # Attach PrintingData for sklearn feature engineering
            if printing_data is not None:
                from dataclasses import replace
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
            artifact = request.app.state.model_artifact
            model = artifact["model"]
            fe = artifact["feature_engineering"]
            sklearn_version = artifact["model_version"]

            X = fe.transform([card])
            log_price = model.predict(X)[0]
            sklearn_price = round(float(np.exp(log_price)), 2)

            # Transformer prediction (optional)
            transformer_result = None
            t_artifact = request.app.state.transformer_artifact
            if t_artifact is not None:
                try:
                    t_model = t_artifact["model"]
                    t_config = t_artifact["config"]
                    t_version = t_artifact["model_version"]

                    tok = request.app.state.tokenizer
                    if tok is None:
                        raise RuntimeError("Tokenizer not loaded — run 'vocabulary' first")

                    shifted_log_pred = predict_shifted_log(
                        t_model, tok, body, printing_data, t_config
                    )
                    t_price = round(
                        shifted_log_to_eur(shifted_log_pred, t_config.log_offset), 2
                    )
                    transformer_result = {
                        "predicted_price_eur": t_price,
                        "model_version": t_version,
                    }
                except Exception as e:
                    logger.warning("Transformer prediction failed: %s", e)

            latency_ms = (time.perf_counter() - start) * 1000
            mana_cost_raw = None
            for line in body.splitlines():
                if line.strip().lower().startswith("mana cost:"):
                    mana_cost_raw = line.split(":", 1)[1].strip() or None
                    break

            log_extra = {
                "card_name": card.name,
                "card_types": list(card.types),
                "card_mana_cost": mana_cost_raw,
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

            return JSONResponse(
                status_code=200,
                content={
                    "sklearn": {
                        "predicted_price_eur": sklearn_price,
                        "model_version": sklearn_version,
                    },
                    "transformer": transformer_result,
                },
            )
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
