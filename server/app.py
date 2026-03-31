# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Dynamic Pricing Env Environment.

This module creates an HTTP server that exposes the DynamicPricingEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from openenv.core.env_server import create_fastapi_app
from models import PricingAction, PricingObservation
from server.dynamic_pricing_env_environment import DynamicPricingEnvironment

app = create_fastapi_app(
    DynamicPricingEnvironment,
    PricingAction,
    PricingObservation,
)
