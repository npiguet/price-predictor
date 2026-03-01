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

from price_predictor.infrastructure.forge_parser import parse_forge_text

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


def create_app(model_artifact: dict[str, Any]) -> FastAPI:
    """Create a FastAPI application with the given model artifact.

    Args:
        model_artifact: Dict with 'model', 'feature_engineering', and 'model_version' keys.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="Price Predictor Service")
    app.state.model_artifact = model_artifact

    @app.post("/api/v1/evaluate")
    async def evaluate(request: Request) -> Response:
        start = time.perf_counter()
        body = (await request.body()).decode("utf-8")

        try:
            card = parse_forge_text(body)
        except (ValueError, TypeError) as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(json.dumps(_build_log_entry(
                status_code=400,
                latency_ms=latency_ms,
                error=str(e),
            )))
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to parse card script: {e}"},
            )

        try:
            artifact = request.app.state.model_artifact
            model = artifact["model"]
            fe = artifact["feature_engineering"]
            model_version = artifact["model_version"]

            X = fe.transform([card])
            log_price = model.predict(X)[0]
            predicted_price = round(float(np.exp(log_price)), 2)

            latency_ms = (time.perf_counter() - start) * 1000
            mana_cost_raw = None
            for line in body.splitlines():
                if line.strip().startswith("ManaCost:"):
                    mana_cost_raw = line.split(":", 1)[1].strip() or None
                    break
            logger.info(json.dumps(_build_log_entry(
                status_code=200,
                latency_ms=latency_ms,
                card_name=card.name,
                card_types=list(card.types),
                card_mana_cost=mana_cost_raw,
                predicted_price_eur=predicted_price,
                model_version=model_version,
            )))

            return JSONResponse(
                status_code=200,
                content={
                    "predicted_price_eur": predicted_price,
                    "model_version": model_version,
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
