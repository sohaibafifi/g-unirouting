"""Dataclasses for decode state and cache (re-exported from mavrp.env.decoders.types)."""
# DecodeCache, DecodeState (+ hidden field), and StepResult are canonical in
# mavrp.env.decoders.types so that both training decoders and XAI share identical types.
from mavrp.env.decoders.types import DecodeCache, DecodeState, StepResult  # noqa: F401

__all__ = ["DecodeCache", "DecodeState", "StepResult"]
