"""Signal registry: signals are plugins.

    from alphacomb.signals.registry import register
    from alphacomb.signals.base import SignalSpec

    @register(SignalSpec(name="MY_SIGNAL", family="alternative", description="...", direction=1,
                         inputs=("vendor.file",), published=2024))
    def my_signal(bundle):
        return frame_with_instrument_timestamp_value

Importing the module registers the signal; experiment configs then refer to it by name. The built-in library lives
in alphacomb.signals.library and is loaded by load_library(); extra modules are loaded with load_plugins().
"""
from __future__ import annotations

import importlib
from typing import Callable, Iterable

from alphacomb.signals.base import FunctionSignal, Signal, SignalSpec

_REGISTRY: dict[str, Signal] = {}


def register(spec: SignalSpec) -> Callable:
    def deco(fn: Callable) -> Callable:
        add(FunctionSignal(spec=spec, fn=fn))
        return fn
    return deco


def add(signal: Signal, replace: bool = False) -> Signal:
    name = signal.spec.name
    if name in _REGISTRY and not replace and _REGISTRY[name].spec != signal.spec:
        raise ValueError(f"signal {name!r} is already registered with a different spec")
    _REGISTRY[name] = signal
    return signal


def get(name: str) -> Signal:
    if name not in _REGISTRY:
        load_library()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown signal {name!r}; registered: {sorted(_REGISTRY)}") from None


def names(family: str | None = None) -> list[str]:
    load_library()
    return sorted(n for n, s in _REGISTRY.items() if family is None or s.spec.family == family)


def specs(selected: Iterable[str] | None = None) -> list[SignalSpec]:
    load_library()
    return [get(n).spec for n in (selected if selected is not None else names())]


def load_plugins(modules: Iterable[str]) -> None:
    for m in modules:
        importlib.import_module(m)


def load_library() -> None:
    importlib.import_module("alphacomb.signals.library")
