"""Trusted startup-only inference adapters; no fitting or calibration in requests."""

from .runtime import ServingRegistry, load_registry

__all__ = ["ServingRegistry", "load_registry"]
