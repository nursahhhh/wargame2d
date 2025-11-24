from __future__ import annotations

import importlib
from typing import Callable, Dict, Type, TypeVar

from .base_agent import BaseAgent

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}
AgentType = TypeVar("AgentType", bound=Type[BaseAgent])


def register_agent(key: str, cls: AgentType | None = None) -> AgentType | Callable[[AgentType], AgentType]:
    """
    Register an agent class under a key for config-based lookup.

    Can be used as a decorator or as a direct call:
    - `@register_agent("foo")` above the class definition
    - `register_agent("foo", FooAgent)` after the class definition (backward compatible)
    """
    def decorator(target_cls: AgentType) -> AgentType:
        AGENT_REGISTRY[key] = target_cls
        return target_cls

    if cls is None:
        return decorator

    return decorator(cls)


def resolve_agent_class(type_ref: str) -> Type[BaseAgent]:
    """
    Resolve an agent class from a registry key or import path.
    
    If `type_ref` matches a registered key, the registry entry is returned.
    Otherwise, the string is treated as a module path like "module.Class".
    """
    if type_ref in AGENT_REGISTRY:
        return AGENT_REGISTRY[type_ref]

    if "." not in type_ref:
        raise ValueError(
            f"Unknown agent type '{type_ref}'. "
            "Provide a registered key or an import path like 'pkg.module.Class'."
        )

    module_name, class_name = type_ref.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    if not issubclass(cls, BaseAgent):
        raise TypeError(f"{type_ref} is not a BaseAgent subclass")

    return cls
