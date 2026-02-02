"""
MiniConf - Minimal YAML configuration with type checking.

Features:
- File inclusion with $() syntax
- Internal references with ${} syntax (dot-notation)
- Type-checked access with clear error messages
- Validation constraints (gt, ge, lt, le, choices)
- Freeze configs to prevent mutation
- @configclass decorator with namespaced config access

Path syntax:
    API Access (Python): Always use "/" separator.
        "/a/b"  -> absolute from root
        "a/b"   -> relative from current section
    
    YAML References: Use "." separator (standard dot-notation).
        ${model.latent_dim}

Usage:
    conf = MiniConf.load("configs/base.yaml").freeze()
    
    @configclass
    class Encoder(nn.Module):
        # 'latent_dim' from conf dict, 'opt/lr' from ns['opt']
        latent_dim: int = config_field("latent_dim", gt=0) 
        lr: float = config_field("opt/lr")                  
        
        def __init__(self, in_features):
            super().__init__()
            self.fc = nn.Linear(in_features, self.latent_dim)
    
    # Initializing with select()
    # Passes conf["model"] and ns["optimizer"]
    encoder = Encoder(128, **conf.select("/model", opt="/optimizer"))
"""

from __future__ import annotations
from typing import Any, Optional, TypeVar, Type, Union, get_origin, get_args, get_type_hints
from pathlib import Path
import yaml
import re
import copy

T = TypeVar("T")

# =============================================================================
# Constants & Patterns
# =============================================================================

INCLUDE_PATTERN = re.compile(r"^\$\(([^)]+)\)$")
REF_PATTERN = re.compile(r"\$\{([^}]+)\}")

# =============================================================================
# Path Utilities
# =============================================================================

def _resolve_path_dot(data: dict, path: str) -> Any:
    """
    Resolve a path using dot notation (e.g., "model.dim").
    Used for internal YAML reference resolution.
    """
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current

def _resolve_path_slash(data: dict, path: str) -> Any:
    """
    Resolve a path using slash notation (e.g., "model/dim").
    Used for API access and configclass lookups.
    """
    current = data
    if not path:
        return current
    for part in path.split("/"):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            available = list(current.keys()) if isinstance(current, dict) else []
            raise KeyError(f"Path '{path}' not found. Available at current level: {available}")
        current = current[part]
    return current

# =============================================================================
# YAML Loading & Resolution
# =============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def _resolve_includes(data: Any, current_dir: Path, config_root: Path, visited: Optional[set[Path]] = None) -> Any:
    """Recursively resolve $(file_path) includes. Detects circular dependencies."""
    if visited is None:
        visited = set()

    if isinstance(data, str):
        match = INCLUDE_PATTERN.fullmatch(data.strip())
        if match:
            return _load_include(match.group(1), current_dir, config_root, visited)
        return data

    if isinstance(data, dict):
        return {k: _resolve_includes(v, current_dir, config_root, visited) for k, v in data.items()}
    
    if isinstance(data, list):
        return [_resolve_includes(item, current_dir, config_root, visited) for item in data]

    return data

def _load_include(include_path: str, current_dir: Path, config_root: Path, visited: set[Path]) -> Any:
    """Load and resolve a single included file."""
    # Determine absolute path
    if include_path.startswith("/"):
        file_path = config_root / include_path[1:]
    else:
        file_path = current_dir / include_path
    
    file_path = file_path.resolve()
    
    # Circular dependency check
    if file_path in visited:
        raise RecursionError(f"Circular dependency detected in config includes: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Included config not found: {file_path}")
    
    with open(file_path) as f:
        included_data = yaml.safe_load(f) or {}
    
    new_visited = visited | {file_path}
    # Recursively resolve includes in the new file
    resolved_data = _resolve_includes(included_data, file_path.parent, config_root, new_visited)
    
    return resolved_data

def load_yaml(path: str | Path, config_root: Optional[Path] = None) -> dict:
    """
    Load a YAML file, resolving includes and references.
    
    Args:
        path: Path to the YAML file.
        config_root: Root directory for absolute includes (default: parent of 'path').
    """
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
        # Check if string is *only* a reference (preserve type)
        match = REF_PATTERN.fullmatch(data.strip())
        if match:
            ref_path = match.group(1)
            value = _resolve_path_dot(root, ref_path)
            if value is None:
                raise ValueError(f"Config reference '${{{ref_path}}}' not found")
            return value
        
        # Check if string contains embedded references (interpolation)
        if REF_PATTERN.search(data):
            def replace_ref(m: re.Match) -> str:
                ref_path = m.group(1)
                value = _resolve_path_dot(root, ref_path)
                if value is None:
                    raise ValueError(f"Config reference '${{{ref_path}}}' not found")
                return str(value)
            return REF_PATTERN.sub(replace_ref, data)
        
        return data

    if isinstance(data, dict):
        return {k: _resolve_refs(v, root) for k, v in data.items()}
    
    if isinstance(data, list):
        return [_resolve_refs(item, root) for item in data]

    return data

# =============================================================================
# Type Checking
# =============================================================================

def _format_type(tp: Type) -> str:
    """Get a readable string representation of a type."""
    origin = get_origin(tp)
    if origin is None:
        return getattr(tp, "__name__", str(tp))
    args = get_args(tp)
    if args:
        args_str = ", ".join(_format_type(a) for a in args)
        return f"{getattr(origin, '__name__', str(origin))}[{args_str}]"
    return str(tp)

def _check_type(value: Any, expected: Type, key: str) -> None:
    """Validate that `value` matches `expected` type. Raises TypeError if not."""
    origin = get_origin(expected)

    # Handle None / Optional
    if value is None:
        # Optional[T] is Union[T, None]
        if origin is Union and type(None) in get_args(expected):
            return
        raise TypeError(f"Config '{key}' is None, expected non-optional {_format_type(expected)}")

    # Handle Union
    if origin is Union:
        non_none_args = [a for a in get_args(expected) if a is not type(None)]
        errors = []
        for arg in non_none_args:
            try:
                _check_type(value, arg, key)
                return # Success
            except TypeError as e:
                errors.append(str(e))
        
        types_str = " | ".join(_format_type(a) for a in non_none_args)
        raise TypeError(
            f"Config '{key}' has wrong type.\n"
            f"  Expected: {types_str}\n"
            f"  Got: {type(value).__name__} ({value!r})"
        )

    # Handle Generic types (List, Dict, etc.)
    if origin is not None:
        if not isinstance(value, origin):
            raise TypeError(
                f"Config '{key}' has wrong type container.\n"
                f"  Expected: {_format_type(expected)}\n"
                f"  Got: {type(value).__name__}"
            )
        args = get_args(expected)
        if origin is list and args:
            for i, elem in enumerate(value):
                _check_type(elem, args[0], f"{key}[{i}]")
        elif origin is dict and len(args) >= 2:
            for k, v in value.items():
                _check_type(k, args[0], f"{key}.keys")
                _check_type(v, args[1], f"{key}.{k}")
        return

    # Handle Primitives
    # Allow int -> float coercion, but not bool -> int or float -> int
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return
    if expected is int and isinstance(value, bool):
        raise TypeError(f"Config '{key}' is bool, expected int")
    
    if not isinstance(value, expected):
        raise TypeError(
            f"Config '{key}' has wrong type.\n"
            f"  Expected: {_format_type(expected)}\n"
            f"  Got: {type(value).__name__} ({value!r})"
        )

# =============================================================================
# MiniConf Core
# =============================================================================

class FrozenDict(dict):
    """Immutable dictionary."""
    __slots__ = ()
    
    def __setitem__(self, key, value): raise TypeError("Config is frozen")
    def __delitem__(self, key): raise TypeError("Config is frozen")
    def clear(self): raise TypeError("Config is frozen")
    def pop(self, *args): raise TypeError("Config is frozen")
    def popitem(self): raise TypeError("Config is frozen")
    def update(self, *args, **kwargs): raise TypeError("Config is frozen")
    def setdefault(self, *args): raise TypeError("Config is frozen")
    def __hash__(self): 
        # Make it hashable so it can be used in sets/dicts
        return hash(tuple(sorted(self.items())))

class FrozenList(list):
    """Immutable list."""
    __slots__ = ()
    def __setitem__(self, index, value): raise TypeError("Config is frozen")
    def __delitem__(self, index): raise TypeError("Config is frozen")
    def append(self, value): raise TypeError("Config is frozen")
    def extend(self, values): raise TypeError("Config is frozen")
    def insert(self, index, value): raise TypeError("Config is frozen")
    def remove(self, value): raise TypeError("Config is frozen")
    def pop(self, *args): raise TypeError("Config is frozen")
    def clear(self): raise TypeError("Config is frozen")

def _freeze_data(data: Any) -> Any:
    """Recursively freeze dicts and lists."""
    if isinstance(data, dict):
        return FrozenDict({k: _freeze_data(v) for k, v in data.items()})
    if isinstance(data, list):
        return FrozenList([_freeze_data(item) for item in data])
    return data

class MiniConf:
    """
    Configuration container with type-checked access and namespace support.
    
    Path syntax:
        - Starts with `/`: Absolute path from root.
        - No leading `/`: Relative path from current section.
    """

    def __init__(self, data: dict | None = None, *, section: str = "", config_path: str | None = None):
        self._root = data or {}
        self._section = section
        self._path = config_path
        self._frozen = False

        # Resolve current section data
        self._data = self._root
        if section:
            self._data = self._navigate(section, from_root=True)

    @classmethod
    def load(cls, path: str | Path, config_root: Path | None = None) -> MiniConf:
        path = Path(path).resolve()
        data = load_yaml(path, config_root)
        return cls(data, config_path=str(path))

    # ---------------------------------------------------------------------
    # Internal Navigation
    # ---------------------------------------------------------------------

    def _navigate(self, path: str, from_root: bool = False) -> Any:
        """Helper to navigate to a path."""
        start = self._root if from_root else self._data
        if not path:
            return start
        
        # Determine if path is absolute (within the context of navigation)
        # If called with from_root=True, relative to root. 
        # If called normally, check leading slash.
        base = self._root if (from_root or path.startswith("/")) else self._data
        lookup_path = path[1:] if (path.startswith("/") and not from_root) else path
        
        return _resolve_path_slash(base, lookup_path)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get(self, key: str, expected_type: Type[T] | None = None) -> T:
        """
        Get config value with optional type checking.
        
        Args:
            key: Path using / separator.
            expected_type: Type to validate against.
        """
        value = self._navigate(key)
        
        if expected_type is not None:
            _check_type(value, expected_type, key)
            
        return value # type: ignore

    def subsection(self, key: str) -> MiniConf:
        """Get a subsection as a new MiniConf instance."""
        new_section = f"{self._section}/{key}" if self._section else key
        # Note: self._navigate will raise if path not found
        return MiniConf(self._root, section=new_section, config_path=self._path)

    def select(self, conf_path: str = "", **ns_paths: str) -> dict:
        """
        Select sections for @configclass unpacking.
        
        Args:
            conf_path: Path for the default 'conf' argument.
            **ns_paths: Mappings for namespaces (e.g., opt="/optimizer").
        
        Returns:
            Dictionary suitable for **unpacking into a @configclass.
        """
        result: dict[str, Any] = {}
        
        # Resolve main conf
        result["conf"] = self._navigate(conf_path)
        
        # Resolve namespaces
        if ns_paths:
            result["ns"] = {name: self._navigate(path) for name, path in ns_paths.items()}
        
        return result
    
    def asdict(self) -> dict:
        """Return the current section as a dictionary (deep copy)."""
        return copy.deepcopy(self._data)

    def freeze(self) -> MiniConf:
        """Make the config immutable."""
        if not self._frozen:
            self._root = _freeze_data(self._root)
            # Refresh _data pointer in case root object identity changed (it did)
            self._data = self._navigate(self._section, from_root=True)
            self._frozen = True
        return self

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def pprint(self, indent: int = 2) -> None:
        print(self.dumps(indent=indent))

    def dumps(self, indent: int = 2) -> str:
        return _format_yaml(self._data, indent=indent)

    def __contains__(self, key: str) -> bool:
        try:
            self._navigate(key)
            return True
        except KeyError:
            return False

    def __repr__(self) -> str:
        section = f"[{self._section}]" if self._section else ""
        path = f" from {self._path}" if self._path else ""
        frozen = " (frozen)" if self._frozen else ""
        return f"MiniConf{section}{path}{frozen}"

# =============================================================================
# YAML Formatting (Custom Dumper)
# =============================================================================

def _format_scalar(value: Any) -> str:
    """Format scalar value to safe YAML string."""
    if value is None: return "null"
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, (int, float)): return str(value)
    if isinstance(value, str):
        # Simple heuristics for quoting
        if "\n" in value or value.startswith(" ") or value.endswith(" "):
            # Use literal block style for multiline
            if "\n" in value: return f"|\n" + "\n".join(f"  {line}" for line in value.split("\n"))
            return repr(value)
        # Avoid numbers/booleans being treated as non-strings
        if value.lower() in ("true", "false", "null", "y", "n", "yes", "no"):
            return f"'{value}'"
        try:
            float(value)
            return f"'{value}'"
        except ValueError:
            pass
        return value
    return str(value)

def _format_yaml(data: Any, indent: int = 2, level: int = 0) -> str:
    """Recursive YAML formatter."""
    prefix = " " * (indent * level)
    
    if isinstance(data, dict):
        if not data: return "{}"
        lines = []
        for key, value in data.items():
            str_key = _format_scalar(key) if isinstance(key, str) else str(key)
            
            if isinstance(value, dict):
                if value:
                    lines.append(f"{prefix}{str_key}:")
                    lines.append(_format_yaml(value, indent, level + 1))
                else:
                    lines.append(f"{prefix}{str_key}: {{}}")
            elif isinstance(value, list):
                if value:
                    lines.append(f"{prefix}{str_key}:")
                    lines.append(_format_yaml(value, indent, level + 1))
                else:
                    lines.append(f"{prefix}{str_key}: []")
            else:
                lines.append(f"{prefix}{str_key}: {_format_scalar(value)}")
        return "\n".join(lines)
    
    if isinstance(data, list):
        if not data: return "[]"
        lines = []
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_format_yaml(item, indent, level + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return "\n".join(lines)
    
    return f"{prefix}{_format_scalar(data)}"


# =============================================================================
# @configclass
# =============================================================================

class _ConfigFieldMarker:
    __slots__ = ("key", "expected_type", "gt", "ge", "lt", "le", "choices")
    
    def __init__(self, key: str, expected_type: Type | None, constraints: dict):
        self.key = key
        self.expected_type = expected_type
        self.gt = constraints.get("gt")
        self.ge = constraints.get("ge")
        self.lt = constraints.get("lt")
        self.le = constraints.get("le")
        self.choices = constraints.get("choices")

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
    Declare a config field.
    
    Args:
        key: Path to value (e.g. "lr" or "opt/lr").
        type: Expected type (inferred from annotation if None).
        **constraints: Validation (gt, ge, lt, le, choices).
    """
    constraints = {k: v for k, v in locals().items() if k in ("gt", "ge", "lt", "le", "choices") and v is not None}
    return _ConfigFieldMarker(key, type, constraints) # type: ignore

def _validate_constraints(value: Any, field: _ConfigFieldMarker, key: str) -> None:
    """Run validation checks defined in config_field."""
    # Only evaluate condition when the constraint is set (not None)
    if field.gt is not None and not (value > field.gt):
        raise ValueError(f"Config '{key}' must be > {field.gt}, got {value}")
    if field.ge is not None and not (value >= field.ge):
        raise ValueError(f"Config '{key}' must be >= {field.ge}, got {value}")
    if field.lt is not None and not (value < field.lt):
        raise ValueError(f"Config '{key}' must be < {field.lt}, got {value}")
    if field.le is not None and not (value <= field.le):
        raise ValueError(f"Config '{key}' must be <= {field.le}, got {value}")
    if field.choices is not None and value not in field.choices:
        raise ValueError(f"Config '{key}' must be one of {field.choices}, got {value!r}")

class ConfigContext:
    """
    Helper to access config within a @configclass.
    """
    def __init__(self, conf: dict | MiniConf, ns: dict | None, root: MiniConf | None):
        self._raw_conf = conf
        self._raw_ns = ns or {}
        self._root = root

    @property
    def conf(self) -> dict:
        return self._raw_conf._data if isinstance(self._raw_conf, MiniConf) else self._raw_conf

    @property
    def ns(self) -> dict[str, dict]:
        result = {}
        for k, v in self._raw_ns.items():
            result[k] = v._data if isinstance(v, MiniConf) else v
        return result

    @property
    def root(self) -> MiniConf | None:
        return self._root

    def select(self, conf_path: str = "", **ns_paths: str) -> dict:
        """Smart selection for child initialization."""
        # 1. Resolve main conf
        if conf_path.startswith("/"):
            if not self._root: raise ValueError("Absolute path requires root")
            c = self._root._navigate(conf_path)
        elif conf_path:
            c = _resolve_path_slash(self.conf, conf_path)
        else:
            c = self.conf
            
        res = {"conf": c}
        
        # 2. Resolve namespaces
        if ns_paths:
            new_ns = {}
            for name, path in ns_paths.items():
                if path.startswith("/"):
                    if not self._root: raise ValueError("Absolute path requires root")
                    new_ns[name] = self._root._navigate(path)
                elif "/" in path and path.split("/")[0] in self._raw_ns:
                    # Path into existing namespace: "opt/adam" -> ns["opt"]["adam"]
                    ns_name, rest = path.split("/", 1)
                    ns_val = self._raw_ns[ns_name]
                    ns_dict = ns_val._data if isinstance(ns_val, MiniConf) else ns_val
                    new_ns[name] = _resolve_path_slash(ns_dict, rest)
                elif path in self._raw_ns:
                    new_ns[name] = self._raw_ns[path]
                else:
                    # Fallback to relative from conf
                    new_ns[name] = _resolve_path_slash(self.conf, path)
            res["ns"] = new_ns
            
        return res

def configclass(cls):
    """
    Decorator to bind config fields to class attributes.
    
    Usage:
        @configclass
        class Model:
            lr: float = config_field("lr")
            
        model = Model(**conf.select("/model"))
    """
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    # Extract markers
    fields: dict[str, _ConfigFieldMarker] = {}
    for name, value in list(vars(cls).items()):
        if isinstance(value, _ConfigFieldMarker):
            if value.expected_type is None:
                value.expected_type = hints.get(name)
            fields[name] = value
            delattr(cls, name)

    if not fields:
        return cls

    cls.__config_fields__ = fields
    original_init = cls.__init__

    def __init__(self, *args, conf: Any = None, ns: Any = None, _from_config_dict: dict | None = None, **kwargs):
        # Path A: Recreated from saved config_dict
        if _from_config_dict is not None:
            for attr, field in fields.items():
                if attr not in _from_config_dict:
                    raise KeyError(f"Missing field '{attr}' in config_dict")
                val = _from_config_dict[attr]
                if field.expected_type: _check_type(val, field.expected_type, attr)
                _validate_constraints(val, field, attr)
                setattr(self, attr, val)
            
            self._conf = _from_config_dict
            self._ns = None
            self._config = ConfigContext(_from_config_dict, None, None)
            original_init(self, *args, **kwargs)
            return

        # Path B: Standard initialization with MiniConf/dict
        if conf is None:
            raise TypeError(f"{cls.__name__}() missing required argument: 'conf'")

        # Resolve Root MiniConf
        root_miniconf = None
        if isinstance(conf, MiniConf):
            root_miniconf = MiniConf(conf._root, config_path=conf._path)
            conf_data = conf._data
        else:
            conf_data = conf

        # Normalize Namespaces
        ns_data: dict[str, dict] = {}
        if ns:
            for k, v in ns.items():
                ns_data[k] = v._data if isinstance(v, MiniConf) else v

        # Populate Fields
        for attr, field in fields.items():
            key = field.key
            
            # Logic: "opt/lr" -> checks ns["opt"]["lr"] first, then conf["opt"]["lr"]
            val = None
            found = False
            
            if "/" in key:
                parts = key.split("/", 1)
                ns_name, rest = parts[0], parts[1]
                
                # Try Namespace
                if ns_name in ns_data:
                    try:
                        val = _resolve_path_slash(ns_data[ns_name], rest)
                        found = True
                    except KeyError:
                        pass
                
                # Fallback to nested Conf
                if not found and isinstance(conf_data, dict) and ns_name in conf_data:
                    try:
                        val = _resolve_path_slash(conf_data[ns_name], rest)
                        found = True
                    except KeyError:
                        pass
            else:
                # Simple key in Conf
                if isinstance(conf_data, dict) and key in conf_data:
                    val = conf_data[key]
                    found = True

            if not found:
                # Construct helpful error
                avail_conf = list(conf_data.keys()) if isinstance(conf_data, dict) else []
                avail_ns = list(ns_data.keys())
                raise KeyError(
                    f"Config key '{key}' not found.\n"
                    f"  Checked in conf: {avail_conf}\n"
                    f"  Checked in namespaces: {avail_ns}"
                )

            if field.expected_type: _check_type(val, field.expected_type, key)
            _validate_constraints(val, field, key)
            setattr(self, attr, val)

        # Set Context Properties
        self._conf = conf
        self._ns = ns
        self._config = ConfigContext(conf_data, ns_data, root_miniconf)
        
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__

    def config_dict(self) -> dict:
        return {attr: getattr(self, attr) for attr in fields}
    
    @classmethod
    def from_config(cls, config: dict, *args, **kwargs):
        return cls(*args, _from_config_dict=config, **kwargs)
    
    cls.config_dict = config_dict
    cls.from_config = from_config

    return cls

__all__ = ["MiniConf", "ConfigContext", "load_yaml", "configclass", "config_field"]