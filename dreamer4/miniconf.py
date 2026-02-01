"""
MiniConf - Minimal YAML configuration with type checking.

Features:
- File inclusion with $() syntax
- Internal references with ${} syntax
- Type-checked access with clear error messages
- Validation constraints (gt, ge, lt, le, choices)
- Freeze configs to prevent mutation
- @configclass decorator with namespaced config access

Path syntax (always use / separator):
    "/a/b"  → absolute from root
    "a/b"   → relative from current section

Usage:
    conf = MiniConf.load("configs/base.yaml").freeze()
    
    @configclass
    class Encoder(nn.Module):
        latent_dim: int = config_field("latent_dim", gt=0)  # from conf
        lr: float = config_field("opt/lr")                  # from ns["opt"]
        
        def __init__(self, in_features):
            super().__init__()
            self.fc = nn.Linear(in_features, self.latent_dim)
    
    # Using select() with / paths
    encoder = Encoder(128, **conf.select("/model", opt="/optimizer"))
    
    # Config stored as self._conf and self._ns

YAML features:
    # Include other files
    model: $(model.yaml)           # relative path
    base: $(/shared/base.yaml)     # absolute from config root
    
    # Reference other values (YAML uses . for nested keys)
    model:
      latent_dim: 64
      decoder_dim: ${model.latent_dim}  # resolves to 64
"""

from __future__ import annotations
from typing import Any, Optional, TypeVar, Type, Union, get_origin, get_args, get_type_hints
from pathlib import Path
import yaml
import re

T = TypeVar("T")

INCLUDE_PATTERN = re.compile(r"\$\(([^)]+)\)")
REF_PATTERN = re.compile(r"\$\{([^}]+)\}")


# =============================================================================
# YAML Loading
# =============================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_includes(data: Any, current_dir: Path, config_root: Path) -> Any:
    if isinstance(data, str):
        match = INCLUDE_PATTERN.fullmatch(data.strip())
        if match:
            return _load_include(match.group(1), current_dir, config_root)
        return data

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                match = INCLUDE_PATTERN.fullmatch(value.strip())
                if match:
                    result[key] = _load_include(match.group(1), current_dir, config_root)
                else:
                    result[key] = value
            else:
                result[key] = _resolve_includes(value, current_dir, config_root)
        return result

    if isinstance(data, list):
        return [_resolve_includes(item, current_dir, config_root) for item in data]

    return data


def _load_include(include_path: str, current_dir: Path, config_root: Path) -> Any:
    if include_path.startswith("/"):
        file_path = config_root / include_path[1:]
    else:
        file_path = current_dir / include_path

    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Included config not found: {file_path}")

    with open(file_path) as f:
        included_data = yaml.safe_load(f) or {}

    return _resolve_includes(included_data, file_path.parent, config_root)


def load_yaml(path: str | Path, config_root: Path | None = None) -> dict:
    path = Path(path).resolve()
    config_root = Path(config_root).resolve() if config_root else path.parent.parent

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    data = _resolve_includes(data, path.parent, config_root)
    data = _resolve_refs(data, data)
    return data


def _resolve_refs(data: Any, root: dict) -> Any:
    """Resolve ${path.to.value} references."""
    if isinstance(data, str):
        # Handle string with embedded references
        def replace_ref(match: re.Match) -> str:
            ref_path = match.group(1)
            value = _get_by_path(root, ref_path)
            if value is None:
                raise ValueError(f"Config reference '${{{ref_path}}}' not found")
            return str(value)
        
        # Check if entire string is a single reference (preserve type)
        match = REF_PATTERN.fullmatch(data.strip())
        if match:
            ref_path = match.group(1)
            value = _get_by_path(root, ref_path)
            if value is None:
                raise ValueError(f"Config reference '${{{ref_path}}}' not found")
            return value
        
        # Otherwise do string interpolation
        if REF_PATTERN.search(data):
            return REF_PATTERN.sub(replace_ref, data)
        return data

    if isinstance(data, dict):
        return {k: _resolve_refs(v, root) for k, v in data.items()}

    if isinstance(data, list):
        return [_resolve_refs(item, root) for item in data]

    return data


def _get_by_path(data: dict, path: str) -> Any:
    """Get nested value by dot-separated path."""
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


# =============================================================================
# Type Checking
# =============================================================================


def _format_type(tp: Type) -> str:
    origin = get_origin(tp)
    if origin is None:
        return getattr(tp, "__name__", str(tp))
    args = get_args(tp)
    if args:
        args_str = ", ".join(_format_type(a) for a in args)
        return f"{getattr(origin, '__name__', str(origin))}[{args_str}]"
    return str(tp)


def _format_value(value: Any, max_len: int = 50) -> str:
    s = repr(value)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _check_type(value: Any, expected: Type, key: str) -> None:
    origin = get_origin(expected)

    if value is None:
        if origin is Union and type(None) in get_args(expected):
            return
        raise TypeError(f"Config '{key}' is None, expected {_format_type(expected)}")

    if origin is Union:
        for arg in get_args(expected):
            if arg is type(None):
                continue
            try:
                _check_type(value, arg, key)
                return
            except TypeError:
                continue
        types = " | ".join(_format_type(a) for a in get_args(expected) if a is not type(None))
        raise TypeError(
            f"Config '{key}' has wrong type\n"
            f"  Expected: {types}\n"
            f"  Got: {type(value).__name__} = {_format_value(value)}"
        )

    if origin is not None:
        if not isinstance(value, origin):
            raise TypeError(
                f"Config '{key}' has wrong type\n"
                f"  Expected: {_format_type(expected)}\n"
                f"  Got: {type(value).__name__} = {_format_value(value)}"
            )
        args = get_args(expected)
        if origin is list and args:
            for i, elem in enumerate(value):
                _check_type(elem, args[0], f"{key}[{i}]")
        elif origin is dict and len(args) >= 2:
            for k, v in value.items():
                _check_type(k, args[0], f"{key}.{{key}}")
                _check_type(v, args[1], f"{key}.{k}")
        return

    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return
    if expected is int and isinstance(value, bool):
        raise TypeError(f"Config '{key}' has wrong type\n  Expected: int\n  Got: bool = {value}")
    if expected is bool and isinstance(value, int) and not isinstance(value, bool):
        raise TypeError(f"Config '{key}' has wrong type\n  Expected: bool\n  Got: int = {value}")

    if not isinstance(value, expected):
        raise TypeError(
            f"Config '{key}' has wrong type\n"
            f"  Expected: {_format_type(expected)}\n"
            f"  Got: {type(value).__name__} = {_format_value(value)}"
        )


# =============================================================================
# MiniConf
# =============================================================================


class MiniConf:
    """
    Configuration container with type-checked access.
    
    Usage:
        config = MiniConf.load("configs/base.yaml")
        lr = config.get("training.lr", float)
    """

    def __init__(self, data: dict | None = None, *, section: str = "", config_path: str | None = None):
        self._root = data or {}
        self._section = section
        self._path = config_path
        self._frozen = False

        self._data = self._root
        if section:
            for part in section.split("/"):
                if not part:
                    continue
                if not isinstance(self._data, dict) or part not in self._data:
                    raise ValueError(f"Config section '{section}' not found")
                self._data = self._data[part]
            if not isinstance(self._data, dict):
                self._data = {}

    @classmethod
    def load(cls, path: str | Path, config_root: Path | None = None) -> MiniConf:
        path = Path(path).resolve()
        data = load_yaml(path, config_root)
        return cls(data, config_path=str(path))

    def get(self, key: str, expected_type: Type[T] | None = None) -> T:
        """
        Get config value with type checking.
        
        Args:
            key: Path using / separator (e.g., "model/latent_dim")
                 Use leading / for absolute path from root
            expected_type: Expected type for validation
        """
        if key.startswith("/"):
            # Absolute from root
            parts = key[1:].split("/") if key != "/" else []
            current = self._root
            full_key = key
        else:
            # Relative from current section
            parts = key.split("/") if key else []
            current = self._data
            full_key = f"{self._section}/{key}" if self._section else key

        for i, part in enumerate(parts):
            if not part:
                continue
            if not isinstance(current, dict):
                raise KeyError(f"Config '{full_key}' not found (not a dict at '{parts[i-1]}')")
            if part not in current:
                available = list(current.keys())
                raise KeyError(f"Config '{full_key}' not found. Available: {available}")
            current = current[part]

        if expected_type is not None:
            _check_type(current, expected_type, full_key)

        return current  # type: ignore

    def subsection(self, key: str) -> MiniConf:
        """Get a subsection as a new MiniConf. Use / for nested paths."""
        new_section = f"{self._section}/{key}" if self._section else key
        instance = MiniConf(self._root, section=new_section, config_path=self._path)
        instance._frozen = self._frozen
        return instance

    def select(self, conf_path: str, **ns_paths: str) -> dict:
        """
        Select a conf section and namespace sections for use with @configclass.
        
        Path syntax (using / separator):
            - "/a/b"  → absolute from root: conf._root["a"]["b"]
            - "a/b"   → relative from current section: conf._data["a"]["b"]
        
        Args:
            conf_path: Path for the default conf
            **ns_paths: namespace_name=path mappings
            
        Returns:
            Dict with 'conf' and 'ns' keys, suitable for **unpacking
            
        Usage:
            conf = MiniConf.load("config.yaml")
            
            # Absolute paths (from root)
            model = Model(**conf.select("/model", opt="/optimizer"))
            
            # Relative paths (from current section)
            sub = conf.subsection("training")
            model = Model(**sub.select("model", opt="optimizer"))
            
            # Mixed
            model = Model(**conf.select("model", opt="/optimizer"))
        """
        result = {"conf": self._resolve_path(conf_path)}
        
        if ns_paths:
            result["ns"] = {name: self._resolve_path(path) for name, path in ns_paths.items()}
        
        return result
    
    def asdict(self, conf_path : Optional[str] = None) -> dict:
        return self._resolve_path(conf_path or "")


    def _resolve_path(self, path: str) -> Any:
        """Resolve a path (absolute with / prefix, or relative)."""
        if path.startswith("/"):
            # Absolute from root
            parts = path[1:].split("/") if path != "/" else []
            current = self._root
        else:
            # Relative from current section
            parts = path.split("/") if path else []
            current = self._data
        
        for part in parts:
            if not part:
                continue
            if not isinstance(current, dict) or part not in current:
                available = list(current.keys()) if isinstance(current, dict) else []
                raise KeyError(f"Path '{path}' not found. Available: {available}")
            current = current[part]
        
        return current

    def freeze(self) -> MiniConf:
        """Make config immutable. Returns self for chaining."""
        self._frozen = True
        self._root = _freeze_dict(self._root)
        self._data = _get_by_path(self._root, self._section) if self._section else self._root
        return self

    def pprint(self, indent: int = 2) -> None:
        """Pretty print config to stdout."""
        print(_format_yaml(self._data, indent=indent))

    def dumps(self, indent: int = 2) -> str:
        """Return config as formatted YAML string."""
        return _format_yaml(self._data, indent=indent)

    @property
    def data(self) -> dict:
        return self._data

    @property
    def root(self) -> dict:
        return self._root

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __repr__(self) -> str:
        section = f"[{self._section}]" if self._section else ""
        path = f" from {self._path}" if self._path else ""
        frozen = " (frozen)" if self._frozen else ""
        return f"MiniConf{section}{path}{frozen}"


class FrozenDict(dict):
    """Immutable dictionary."""
    def __setitem__(self, key, value):
        raise TypeError("Config is frozen")
    def __delitem__(self, key):
        raise TypeError("Config is frozen")
    def update(self, *args, **kwargs):
        raise TypeError("Config is frozen")
    def pop(self, *args):
        raise TypeError("Config is frozen")
    def popitem(self):
        raise TypeError("Config is frozen")
    def clear(self):
        raise TypeError("Config is frozen")
    def setdefault(self, *args):
        raise TypeError("Config is frozen")


class FrozenList(list):
    """Immutable list."""
    def __setitem__(self, index, value):
        raise TypeError("Config is frozen")
    def __delitem__(self, index):
        raise TypeError("Config is frozen")
    def append(self, value):
        raise TypeError("Config is frozen")
    def extend(self, values):
        raise TypeError("Config is frozen")
    def insert(self, index, value):
        raise TypeError("Config is frozen")
    def remove(self, value):
        raise TypeError("Config is frozen")
    def pop(self, *args):
        raise TypeError("Config is frozen")
    def clear(self):
        raise TypeError("Config is frozen")


def _freeze_dict(data: Any) -> Any:
    """Recursively freeze dicts and lists."""
    if isinstance(data, dict):
        return FrozenDict({k: _freeze_dict(v) for k, v in data.items()})
    if isinstance(data, list):
        return FrozenList([_freeze_dict(item) for item in data])
    return data


def _format_yaml(data: Any, indent: int = 2, level: int = 0) -> str:
    """Format data as YAML-like string."""
    prefix = " " * (indent * level)
    
    if isinstance(data, dict):
        if not data:
            return "{}"
        lines = []
        for key, value in data.items():
            if isinstance(value, dict) and value:
                lines.append(f"{prefix}{key}:")
                lines.append(_format_yaml(value, indent, level + 1))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    lines.append(f"{prefix}  -")
                    lines.append(_format_yaml(item, indent, level + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(value)}")
        return "\n".join(lines)
    
    if isinstance(data, list):
        if not data:
            return "[]"
        lines = [f"{prefix}- {_format_scalar(item)}" for item in data]
        return "\n".join(lines)
    
    return f"{prefix}{_format_scalar(data)}"


def _format_scalar(value: Any) -> str:
    """Format scalar value for display."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if "\n" in value or ":" in value or value.startswith(" "):
            return repr(value)
        return value
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(v) for v in value) + "]"
    return str(value)


# =============================================================================
# @configclass
# =============================================================================


class _ConfigFieldMarker:
    __slots__ = ("key", "expected_type", "gt", "ge", "lt", "le", "choices")
    
    def __init__(
        self,
        key: str,
        expected_type: Type | None,
        gt: float | int | None,
        ge: float | int | None,
        lt: float | int | None,
        le: float | int | None,
        choices: list | tuple | None,
    ):
        self.key = key
        self.expected_type = expected_type
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le
        self.choices = choices


def config_field(
    key: str,
    type: Type[T] | None = None,
    *,
    gt: float | int | None = None,
    ge: float | int | None = None,
    lt: float | int | None = None,
    le: float | int | None = None,
    choices: list | tuple | None = None,
) -> T:
    """
    Declare a required config field with optional validation.
    
    Args:
        key: Config key using / separator. Simple keys from default conf.
             Prefix with namespace/ to access from ns (e.g., "opt/lr").
        type: Expected type (usually inferred from annotation)
        gt: Value must be greater than this
        ge: Value must be greater than or equal to this
        lt: Value must be less than this
        le: Value must be less than or equal to this
        choices: Value must be one of these options
        
    Examples:
        latent_dim: int = config_field("latent_dim", gt=0)      # from conf
        lr: float = config_field("opt/lr")                      # from ns["opt"]
        beta1: float = config_field("opt/adam/beta1")           # ns["opt"]["adam"]["beta1"]
    """
    return _ConfigFieldMarker(key, type, gt, ge, lt, le, choices)  # type: ignore


def _validate_constraints(value: Any, field: _ConfigFieldMarker, full_key: str) -> None:
    """Validate field constraints."""
    if field.gt is not None and not value > field.gt:
        raise ValueError(f"Config '{full_key}' must be > {field.gt}, got {value}")
    if field.ge is not None and not value >= field.ge:
        raise ValueError(f"Config '{full_key}' must be >= {field.ge}, got {value}")
    if field.lt is not None and not value < field.lt:
        raise ValueError(f"Config '{full_key}' must be < {field.lt}, got {value}")
    if field.le is not None and not value <= field.le:
        raise ValueError(f"Config '{full_key}' must be <= {field.le}, got {value}")
    if field.choices is not None and value not in field.choices:
        raise ValueError(f"Config '{full_key}' must be one of {list(field.choices)}, got {value!r}")


def configclass(cls):
    """
    Decorator for classes that read config fields.
    
    Pass `conf` (default config) and optional `ns` (namespaces) at instantiation.
    These are intercepted by the decorator - no need to include them in __init__.
    Fields are populated and validated BEFORE __init__ runs.
    
    Key syntax (using / separator):
        - "lr"          → from default conf
        - "opt/lr"      → from namespace "opt"
        - "opt/adam/b1" → from ns["opt"]["adam"]["b1"]
    
    Usage:
        @configclass
        class Model(nn.Module):
            latent_dim: int = config_field("latent_dim", gt=0)  # from conf
            lr: float = config_field("opt/lr")                  # from ns["opt"]
            
            def __init__(self, in_features):
                super().__init__()
                self.fc = nn.Linear(in_features, self.latent_dim)
        
        # Using select() with / paths
        model = Model(128, **conf.select("/model", opt="/optimizer"))
        
        # Access stored config via self._conf and self._ns
    """
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    # Collect fields: attr_name -> _ConfigFieldMarker (with type filled in)
    fields: dict[str, _ConfigFieldMarker] = {}

    for name in list(vars(cls)):
        value = getattr(cls, name)
        if isinstance(value, _ConfigFieldMarker):
            if value.expected_type is None:
                value.expected_type = hints.get(name)
            fields[name] = value
            delattr(cls, name)

    if not fields:
        return cls

    cls.__config_fields__ = fields
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        # Extract conf and ns from kwargs (don't pass to original __init__)
        conf = kwargs.pop("conf")
        ns: dict[str, dict] | None = kwargs.pop("ns", None)
        
        # Normalize conf
        if isinstance(conf, MiniConf):
            conf_data = conf.data
        else:
            conf_data = conf
        
        # Normalize namespaces
        ns_data: dict[str, dict] = {}
        if ns:
            for name, val in ns.items():
                if isinstance(val, MiniConf):
                    ns_data[name] = val.data
                else:
                    ns_data[name] = val

        # Populate and validate fields
        for attr, field in fields.items():
            key = field.key
            
            if "/" in key:
                # Try namespaced first: "opt/lr" → ns["opt"]["lr"]
                ns_name, rest = key.split("/", 1)
                if ns_data and ns_name in ns_data:
                    value = _get_nested(ns_data[ns_name], rest, key, ns_name)
                elif isinstance(conf_data, dict) and ns_name in conf_data:
                    # Fall back to nested path in conf: "optim/lr" → conf["optim"]["lr"]
                    value = _get_nested(conf_data[ns_name], rest, key, "conf")
                else:
                    available_keys = list(conf_data.keys()) if isinstance(conf_data, dict) else []
                    available_ns = list(ns_data.keys()) if ns_data else []
                    raise KeyError(
                        f"Config key '{key}' not found\n"
                        f"  conf keys: {available_keys}\n"
                        f"  namespaces: {available_ns}"
                    )
            else:
                # Default conf
                if not isinstance(conf_data, dict) or key not in conf_data:
                    available_keys = list(conf_data.keys()) if isinstance(conf_data, dict) else []
                    available_ns = list(ns_data.keys()) if ns_data else []
                    raise KeyError(
                        f"Config key '{key}' not found in conf\n"
                        f"  conf keys: {available_keys}\n"
                        f"  namespaces: {available_ns}"
                    )
                value = conf_data[key]
            
            if field.expected_type is not None:
                _check_type(value, field.expected_type, key)
            _validate_constraints(value, field, key)
            setattr(self, attr, value)

        self._conf = conf
        self._ns = ns
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


def _get_nested(data: dict, key: str, full_key: str, ns_name: str) -> Any:
    """Get nested value from dict using / separated key."""
    current = data
    parts = [p for p in key.split("/") if p]
    for i, part in enumerate(parts):
        if not isinstance(current, dict):
            path = "/".join(parts[:i])
            raise KeyError(
                f"Config key '{full_key}' not found\n"
                f"  '{path}' in namespace '{ns_name}' is not a dict"
            )
        if part not in current:
            available = list(current.keys())
            path = "/".join(parts[:i]) if i > 0 else ns_name
            raise KeyError(
                f"Config key '{full_key}' not found\n"
                f"  Available at '{path}': {available}"
            )
        current = current[part]
    return current


__all__ = ["MiniConf", "load_yaml", "configclass", "config_field"]