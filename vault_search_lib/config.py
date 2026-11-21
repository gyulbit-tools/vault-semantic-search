"""Configuration: a JSON object, every key optional.

Lookup order for the file: --config, then the VAULT_SEARCH_CONFIG environment
variable, then `.vault-search.json` in the vault root, then built-in defaults.
VAULT_SEARCH_MODEL_DIR and VAULT_SEARCH_CACHE_DIR override `model_dir` and
`cache_dir` from any of these.

A file inside the vault (by its given or its real path) is opened only when no
link or junction is on its path and it is a regular file with one name
(pages.read_file); otherwise the run stops with CONFIG_UNREADABLE and a fixed
reason. Only a missing `.vault-search.json` means defaults. An empty --config
value is refused the same way; it never means the vault's own file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import pages
from .errors import SearchError

CONFIG_FILE_NAME = ".vault-search.json"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"


@dataclass(frozen=True)
class HideRule:
    """Hide snippets and section titles of a page by a front matter field.

    values: hide when the field has one of these values.
    allow:  hide unless the field has one of these values (a missing field hides).
    Rule values and note values compare in one form: trimmed, a value of
    several lines joined with spaces, NFC and casefold.
    """
    field: str
    values: tuple = ()
    allow: tuple | None = None


@dataclass
class Config:
    model_dir: str | None = None
    model_name: str = DEFAULT_MODEL
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    cache_dir: str | None = None
    skip_dot_dirs: bool = True
    skip_dir_names: list = field(default_factory=lambda: [".git", ".trash", "node_modules"])
    exclude_dirs: list = field(default_factory=list)
    private_dirs: list = field(default_factory=list)
    hide_snippets_when: list = field(default_factory=list)
    term_fields: list = field(default_factory=lambda: ["aliases", "alias", "keywords"])
    pin_field: str | None = None
    label_fields: list = field(default_factory=list)
    require_scope_for_paths: bool = False


_STR_LISTS = ("skip_dir_names", "exclude_dirs", "private_dirs", "term_fields", "label_fields")
_BOOLS = ("skip_dot_dirs", "require_scope_for_paths")
_OPTIONAL_STRS = ("model_dir", "cache_dir", "pin_field")
_STRS = ("model_name", "query_prefix", "passage_prefix")


def _is_str_list(value):
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def _hide_rule(raw):
    if not isinstance(raw, dict) or set(raw) - {"field", "values", "allow"}:
        raise SearchError("CONFIG_INVALID", "hide_snippets_when")
    name = raw.get("field")
    if not isinstance(name, str) or not name.strip():
        raise SearchError("CONFIG_INVALID", "hide_snippets_when.field")
    has_values, has_allow = "values" in raw, "allow" in raw
    if has_values == has_allow:
        raise SearchError("CONFIG_INVALID", "hide_snippets_when needs exactly one of values, allow")
    items = raw["values"] if has_values else raw["allow"]
    if not _is_str_list(items) or (has_values and not items):
        raise SearchError("CONFIG_INVALID", "hide_snippets_when." + ("values" if has_values else "allow"))
    if has_values:
        return HideRule(name.strip(), tuple(items), None)
    return HideRule(name.strip(), (), tuple(items))


def from_dict(data):
    if not isinstance(data, dict):
        raise SearchError("CONFIG_INVALID", "top level must be a JSON object")
    known = set(Config.__dataclass_fields__)
    unknown = sorted(set(data) - known)
    if unknown:
        raise SearchError("CONFIG_INVALID", "unknown key " + unknown[0])
    cfg = Config()
    for key, value in data.items():
        if key in _STR_LISTS:
            if not _is_str_list(value):
                raise SearchError("CONFIG_INVALID", key)
        elif key in _BOOLS:
            if type(value) is not bool:
                raise SearchError("CONFIG_INVALID", key)
        elif key in _OPTIONAL_STRS:
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SearchError("CONFIG_INVALID", key)
        elif key in _STRS:
            if not isinstance(value, str) or (key == "model_name" and not value.strip()):
                raise SearchError("CONFIG_INVALID", key)
        elif key == "hide_snippets_when":
            if not isinstance(value, list):
                raise SearchError("CONFIG_INVALID", key)
            value = [_hide_rule(item) for item in value]
        setattr(cfg, key, value)
    return cfg


def load(vault=None, path=None, environ=None):
    """Read the configuration. A missing default file means defaults; a named file must exist.

    vault is the checked vault root (pages.checked_root). An empty or blank
    path (given, not None) is CONFIG_UNREADABLE with a fixed reason.
    """
    environ = os.environ if environ is None else environ
    if path is not None and not str(path).strip():
        # A named but empty path (`--config ""`, an empty variable in a script) never falls back to the
        # vault's own file: the run was meant to use another configuration.
        raise SearchError("CONFIG_UNREADABLE", "empty configuration path")
    chosen = path if path is not None else environ.get("VAULT_SEARCH_CONFIG") or None
    data = {}
    try:
        if chosen is not None:
            if vault is not None:
                raw = pages.read_file(vault, chosen)
            else:
                with open(os.path.expanduser(chosen), "rb") as stream:
                    raw = stream.read()
        elif vault is not None:
            try:
                raw = pages.read_file(vault, os.path.join(vault, CONFIG_FILE_NAME))
            except FileNotFoundError:
                raw = None
        else:
            raw = None
        if raw is not None:
            data = json.loads(raw.decode("utf-8-sig"))
    except SearchError as exc:
        raise SearchError("CONFIG_UNREADABLE", pages.refused_reason(exc.code)) from None
    except OSError:
        raise SearchError("CONFIG_UNREADABLE") from None
    except ValueError:
        raise SearchError("CONFIG_INVALID", "not valid JSON") from None
    cfg = from_dict(data)
    if environ.get("VAULT_SEARCH_MODEL_DIR"):
        cfg.model_dir = environ["VAULT_SEARCH_MODEL_DIR"]
    if environ.get("VAULT_SEARCH_CACHE_DIR"):
        cfg.cache_dir = environ["VAULT_SEARCH_CACHE_DIR"]
    return cfg
