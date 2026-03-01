"""FastAPI prediction service for card price estimation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from price_predictor.infrastructure.forge_parser import parse_forge_text

logger = logging.getLogger(__name__)


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
        body = (await request.body()).decode("utf-8")

        try:
            card = parse_forge_text(body)
        except (ValueError, TypeError) as e:
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

            return JSONResponse(
                status_code=200,
                content={
                    "predicted_price_eur": predicted_price,
                    "model_version": model_version,
                },
            )
        except Exception as e:
            logger.exception("Prediction failed")
            return JSONResponse(
                status_code=500,
                content={"error": f"Prediction failed: {e}"},
            )

    return app
