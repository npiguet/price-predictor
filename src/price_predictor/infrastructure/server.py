"""FastAPI prediction service for card price estimation."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from price_predictor.application.card_enrichment import (
    enrich_or_default,
    extract_printing_data_from_text,
)
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
    app.state.tokenizer = tokenizer

    @app.post("/api/v1/predict")
    async def predict(request: Request) -> Response:
        start = time.perf_counter()
        body = (await request.body()).decode("utf-8")

        # Enrich card text with printing data (auto-fill/default)
        enriched_body = enrich_or_default(body, request.app.state.metadata_map)

        try:
            card = parse_converted_text(enriched_body)
            # Attach printing data to Card for sklearn feature engineering
            pd = extract_printing_data_from_text(enriched_body)
            if pd is not None:
                from dataclasses import replace
                card = replace(card, printing_data=pd)
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
                    import torch

                    t_model = t_artifact["model"]
                    t_config = t_artifact["config"]
                    t_version = t_artifact["model_version"]

                    tok = request.app.state.tokenizer
                    if tok is None:
                        raise RuntimeError("Tokenizer not loaded — run 'vocabulary' first")

                    input_ids_list, attention_mask_list = tok.encode(
                        enriched_body, t_config.max_seq_len
                    )

                    try:
                        device = next(t_model.parameters()).device
                    except StopIteration:
                        device = torch.device("cpu")

                    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
                    attention_mask = torch.tensor(
                        [attention_mask_list], dtype=torch.long
                    ).to(device)

                    t_model.eval()
                    with torch.no_grad():
                        shifted_log_pred = t_model(input_ids, attention_mask).item()

                    t_price = round(float(math.exp(shifted_log_pred) - 2), 2)
                    t_price = max(t_price, 0.0)
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
