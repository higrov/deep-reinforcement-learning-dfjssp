from typing import Any, Dict
import sys

# Try to import TOML parser (Python 3.11+ has tomllib built-in)
try:
    if sys.version_info >= (3, 11):
        import tomllib as toml_parser
        def _load_toml(file_obj):
            return toml_parser.load(file_obj)
    else:
        import tomli as toml_parser  # type: ignore
        def _load_toml(file_obj):
            return toml_parser.load(file_obj)
except ImportError:
    # Fallback: basic TOML-like parser for demo purposes
    def _load_toml(file_obj):
        content = file_obj.read().decode('utf-8')
        return _parse_simple_toml(content)


def load_parameters(toml_path: str) -> Dict[str, Any]:
    """Load parameters from a TOML file.

    This mirrors the minimal interface used in JSBE and keeps the rest of the
    codebase decoupled from the config format.
    """
    with open(toml_path, "rb") as f:
        return _load_toml(f)


def _parse_simple_toml(content: str) -> Dict[str, Any]:
    """Simple TOML-like parser for basic configs (fallback only).
    
    This is a minimal implementation for demo purposes when tomli is not available.
    """
    result = {}
    current_section = result
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Handle sections like [method] or [method.config]
        if line.startswith('[') and line.endswith(']'):
            section_name = line[1:-1]
            # Support nested sections like [method.config]
            parts = section_name.split('.')
            cursor = result
            for part in parts:
                if part not in cursor:
                    cursor[part] = {}
                cursor = cursor[part]
            current_section = cursor
            continue
        
        # Handle key = value pairs
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Strip trailing comments
            if '#' in value:
                value = value.split('#')[0].strip()
            
            # Remove quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # Convert ints/floats where possible
            else:
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
            
            current_section[key] = value
    
    return result
