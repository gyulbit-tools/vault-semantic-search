"""Local ONNX encoder and the embedding cache.

The encoder loads `model.onnx` and `tokenizer.json` from one folder on disk and
never downloads anything. Pooling is the attention-mask mean followed by L2
normalisation, as in the sentence-transformers export of the E5 models.

The cache holds one float16 vector per embedding input, keyed by SHA-1 of the
input text, and remembers the model name. Change `model_name` whenever you put
a different model in `model_dir`; otherwise old vectors are reused.

A cache file is used only when it has this format version, the same model
name, one unique key per vector row, finite unit-length vectors and a SHA-256
digest over the model name, keys and vectors (so rows that no longer belong to
their keys are caught). Any other file is ignored and rebuilt. A usable file
whose vector width is not the model's (same model name, other model files)
stops the run with SEARCH_FAILED before the file is changed.

Runs that save the same cache take turns through a lock file next to it
(`search.npz.lock`), and each save keeps the rows another run saved meanwhile.
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import re
import tempfile
import time
from pathlib import Path

from .errors import SearchError

BACKEND_MODULES = ("numpy", "onnxruntime", "tokenizers")
CACHE_FOLDER = "vault-semantic-search"
CACHE_FORMAT = 2
_KEY_RE = re.compile(r"[0-9a-f]{40}")


class CountingTokenizer:
    """The encode(text, add_special_tokens, truncation=False) call the window builder needs."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text, add_special_tokens=True, truncation=False):
        if truncation:
            raise ValueError("counting tokenizer never truncates")
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


class OnnxEncoder:
    """tokenizer, max_seq_length and encode(), the part of a sentence-transformers model search uses.

    XLM-RoBERTa tokenizers (the E5 multilingual models) are exported with a
    Precompiled normalizer followed by Replace and Metaspace only, which keeps a
    trailing space, newline or zero-width space as a '▁' token. To match the
    Python transformers tokenizer, a Precompiled normalizer is kept alone and the
    pre-tokenizer becomes WhitespaceSplit + Metaspace(always, split). Other
    tokenizers are used as exported. token_type_ids, when the model takes them,
    are zeros.
    """
    max_seq_length = 512

    def __init__(self, directory):
        try:
            import numpy
            import onnxruntime
            from tokenizers import Tokenizer, normalizers, pre_tokenizers
        except (ImportError, OSError):
            raise SearchError("SEARCH_DEPENDENCY_MISSING") from None
        directory = Path(directory)
        tokenizer_file, model_file = directory / "tokenizer.json", directory / "model.onnx"
        if not tokenizer_file.is_file() or not model_file.is_file():
            raise SearchError("SEARCH_MODEL_UNAVAILABLE")
        tokenizer = Tokenizer.from_buffer(tokenizer_file.read_bytes())
        normalizer = tokenizer.normalizer
        if isinstance(normalizer, normalizers.Sequence):
            normalizer = normalizer[0]
        if isinstance(normalizer, normalizers.Precompiled):
            tokenizer.normalizer = normalizer
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)])
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._np, self._raw_tokenizer = numpy, tokenizer
        self.tokenizer = CountingTokenizer(tokenizer)
        pad = tokenizer.token_to_id("<pad>")
        self._pad_id = pad if pad is not None else 0  # masked out, so the value does not matter
        self._session = onnxruntime.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self._inputs = {item.name for item in self._session.get_inputs()}
        outputs = [item.name for item in self._session.get_outputs()]
        if not {"input_ids", "attention_mask"} <= self._inputs or not outputs:
            raise SearchError("SEARCH_MODEL_UNAVAILABLE")
        self._output = "last_hidden_state" if "last_hidden_state" in outputs else outputs[0]
        # Vector width from the output shape ([batch, tokens, width]); None when the export leaves it open.
        shape = next(item.shape for item in self._session.get_outputs() if item.name == self._output)
        self.dimension = shape[-1] if shape and type(shape[-1]) is int and shape[-1] > 0 else None

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False,
               convert_to_numpy=True):
        np, limit = self._np, self.max_seq_length
        texts = list(texts)
        out = np.zeros((len(texts), 0), dtype=np.float32)
        # Longest first, as sentence-transformers does; rows go back to the input order.
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for start in range(0, len(order), batch_size):
            part = order[start:start + batch_size]
            rows = [encoding.ids for encoding in self._raw_tokenizer.encode_batch([texts[i] for i in part])]
            # Same as longest_first truncation for one sequence: first limit-1 tokens + the end token.
            rows = [ids if len(ids) <= limit else ids[:limit - 1] + ids[-1:] for ids in rows]
            width = max(len(ids) for ids in rows)
            ids = np.full((len(rows), width), self._pad_id, dtype=np.int64)
            mask = np.zeros((len(rows), width), dtype=np.int64)
            for row, values in enumerate(rows):
                ids[row, :len(values)] = values
                mask[row, :len(values)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run([self._output], feed)[0]
            weights = mask[:, :, None].astype(np.float32)
            vectors = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            if normalize_embeddings:
                vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
            if out.shape[1] == 0:
                out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
            out[part] = vectors
        return out


def load_encoder(cfg):
    """The configured local model. Missing settings or files are fixed codes, never a download."""
    if not cfg.model_dir:
        raise SearchError("SEARCH_MODEL_NOT_CONFIGURED")
    try:
        return OnnxEncoder(Path(cfg.model_dir).expanduser())
    except SearchError:
        raise
    except Exception:
        raise SearchError("SEARCH_MODEL_UNAVAILABLE") from None


def cache_root(cfg, environ=None):
    """The folder that holds the caches. OSError when it needs the home folder and that cannot be
    determined (`~` in cache_dir, or no cache variable set): a cache that cannot be used, like
    one that cannot be written."""
    environ = os.environ if environ is None else environ
    try:
        if cfg.cache_dir:
            return Path(cfg.cache_dir).expanduser()
        base = environ.get("LOCALAPPDATA") or environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    except RuntimeError:  # Path.expanduser() and Path.home() without a home folder
        raise OSError(errno.ENOENT, "home folder not found for the cache folder") from None
    return Path(base) / CACHE_FOLDER


def cache_file(vault, cfg, environ=None):
    """One cache per vault, outside the vault (so a synced vault does not upload it).

    vault is the checked root (pages.checked_root), spelled as the file system
    spells it (on POSIX, each folder name as its parent lists it), so two
    folders whose names differ only in case or Unicode form get separate caches
    and two spellings of one folder share one, also on a volume that ignores
    case, except that a component keeps its given spelling when its parent's
    listing cannot be read. OSError when the cache folder cannot be named (see cache_root).
    """
    key = hashlib.sha256(os.fspath(vault).encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return cache_root(cfg, environ) / key / "search.npz"


def _digest(np, model_name, keys, vecs):
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8", "surrogatepass") + b"\0")
    h.update(repr(tuple(vecs.shape)).encode("ascii") + b"\0")
    h.update(np.ascontiguousarray(keys, dtype="S40").tobytes())
    h.update(np.ascontiguousarray(vecs, dtype="<f2").tobytes())
    return h.hexdigest()


ABSENT, DISCARD, USABLE = "absent", "discard", "usable"
_ITEMS = ("format", "model", "keys", "vecs", "digest")


def unit_rows(np, vecs):
    """True when every row of the float16 array vecs is finite and of unit length.

    The one rule for vectors read from the cache and for vectors a model has
    just computed, so a vector is never saved or ranked that a later read
    would refuse.
    """
    values = vecs.astype("float32")
    return bool(np.isfinite(values).all() and (not len(values) or
                                               np.abs(np.linalg.norm(values, axis=1) - 1).max() <= 1e-2))


def checked_vectors(np, vectors, count, width=None):
    """The float16 rows of a model output, or SEARCH_FAILED.

    The output must be count rows of width values (of any width above 0 when
    width is None), each finite and of unit length (unit_rows). Used for note
    windows and for questions alike.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(vectors) != count or (width is not None and vectors.shape[1] != width):
        raise SearchError("SEARCH_FAILED")
    vectors = vectors.astype("float16")
    if not vectors.shape[1] or not unit_rows(np, vectors):
        raise SearchError("SEARCH_FAILED")  # a model output a cache read would refuse
    return vectors


def _load(path, model_name):
    """({key: float16 vector}, state) of the cache file.

    state: ABSENT (no file), DISCARD (a file that is unreadable, of another
    format or model, or fails a check: it is replaced on the next save, with no
    vectors), USABLE (every check passed). The format is the integer scalar 2,
    as written; an item that cannot be read as an array (NumPy returns the raw
    bytes of a damaged one) or whose content cannot be read (a model name or
    digest that is not valid text) is a damaged file.
    """
    import numpy as np
    if path is None:  # the cache folder cannot be named (see embed)
        return {}, DISCARD
    try:
        with np.load(path, allow_pickle=False) as data:
            fmt, model, keys, vecs, digest = (data[name] for name in _ITEMS)
    except FileNotFoundError:
        return {}, ABSENT
    except Exception:  # old format, truncated, not an npz file, or a path that cannot be opened
        return {}, DISCARD
    if not all(isinstance(item, np.ndarray) for item in (fmt, model, keys, vecs, digest)):
        return {}, DISCARD
    try:
        if (fmt.shape != () or fmt.dtype.kind not in "iu" or int(fmt) != CACHE_FORMAT
                or model.shape != () or model.dtype.kind != "U" or str(model) != model_name
                or digest.shape != () or digest.dtype.kind != "U"):
            return {}, DISCARD
        if (keys.ndim != 1 or keys.dtype != np.dtype("S40") or vecs.ndim != 2 or vecs.dtype != np.float16
                or len(keys) != len(vecs) or (len(keys) and vecs.shape[1] == 0)):
            return {}, DISCARD
        if str(digest) != _digest(np, model_name, keys, vecs):
            return {}, DISCARD
        names = keys.astype(str).tolist()  # bytes that are not ASCII raise here
        if len(set(names)) != len(names) or not all(_KEY_RE.fullmatch(name) for name in names):
            return {}, DISCARD
        if not unit_rows(np, vecs):
            return {}, DISCARD
    # The content of an item can be damaged as well: key bytes that are not ASCII raise UnicodeError, and
    # a text item that holds a code point above U+10FFFF raises SystemError when NumPy makes it a str.
    except Exception:
        return {}, DISCARD
    return dict(zip(names, vecs)), USABLE


def model_dimension(model):
    """The vector width a model declares (`dimension`), or None."""
    value = getattr(model, "dimension", None)
    return value if type(value) is int and value > 0 else None


LOCK_TIMEOUT = 30.0  # seconds a save waits for another run's save of the same cache
SAVE_WARNING = "warning: could not save the embedding cache; it will be computed again next time"


@contextlib.contextmanager
def _locked(path):
    """Hold the cache's lock file (`search.npz.lock`) so two runs never save the same cache at once.

    One byte of the lock file is locked with msvcrt on Windows and flock
    elsewhere (standard library only). A lock not got within LOCK_TIMEOUT is an
    OSError, which every caller treats as a failed save, as is a cache folder
    that cannot be named (path None).
    """
    if path is None:
        raise OSError(errno.ENOENT, "cache folder not found")
    if "\x00" in os.fspath(path):  # a cache_dir with NUL is a cache write failure, like any other
        raise OSError(errno.EINVAL, "invalid cache path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0),
                         0o666)
    try:
        deadline = time.monotonic() + LOCK_TIMEOUT
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _save(path, model_name, keys, vectors):
    import numpy as np
    if "\x00" in os.fspath(path):  # a cache_dir with NUL is a cache write failure, like any other
        raise OSError(errno.EINVAL, "invalid cache path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    keys = np.array(keys, dtype="S40")
    vecs = np.stack(vectors).astype("float16") if vectors else np.zeros((0, 0), "float16")
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, format=np.array(CACHE_FORMAT), model=np.array(model_name), keys=keys, vecs=vecs,
                     digest=np.array(_digest(np, model_name, keys, vecs)))
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def embed(path, model_name, texts, get_model, prune=True, step=5000, log=None, require_save=False, warn=None):
    """Vectors for texts, computed once and reused from the cache.

    prune: drop cached vectors not used by this run (only for a complete
    whole-vault scan; any other run keeps them). With no texts and prune, an
    existing cache file becomes an empty valid one and no model is needed; no
    file is created where there was none. Progress is saved every `step` new
    inputs, so an interrupted run continues where it stopped. A text that occurs
    several times is computed once. Before any vector is reused or saved, the
    model's vector width is compared with the cache's: the model's `dimension`
    when it has one, else the width of one input it embeds (an input that needs
    embedding anyway, when there is one). Another width is SEARCH_FAILED before
    the cache file is changed. Returns (vectors, new_count): new_count is the
    number of inputs actually computed. A computed vector that is not finite
    and of unit length (the rule a cache read uses) is SEARCH_FAILED before it
    is saved or ranked; progress saved before it is kept. require_save: a
    failed final save is SEARCH_CACHE_WRITE_FAILED instead of a warning (warm
    exists to fill the cache). The warning goes to warn (or log, when warn is
    not given). path None: the cache folder cannot be named (cache_file raised
    OSError); no cache is read and every save fails like a cache that cannot
    be written, so search goes on in memory and warm stops.

    Each save holds the cache's lock (_locked), reads the newest file again and
    adds this run's rows to the rows another run saved meanwhile; only the
    final save of a prune run leaves out the rows this run does not use. The
    model runs outside the lock.
    """
    import numpy as np
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    known, state = _load(path, model_name)
    cached = set(known)
    warn = warn or log
    width = None

    def merged(final):
        """The rows to write, or None when the newest file already holds them."""
        latest, latest_state = _load(path, model_name)
        if latest_state != USABLE:
            latest = {}
        if latest and width is not None and len(next(iter(latest.values()))) != width:
            raise SearchError("SEARCH_FAILED")  # another run saved vectors of another width meanwhile
        mine = {k: known[k] for k in dict.fromkeys(keys) if k in known}
        rows = mine if final and prune else {**latest, **mine}
        if latest_state == USABLE and rows.keys() == latest.keys():
            return None
        if latest_state == ABSENT and not rows:
            return None  # no file is made for a run that has nothing to keep
        return rows

    def save(final=False):
        try:
            with _locked(path):
                rows = merged(final)
                if rows is not None:
                    _save(path, model_name, list(rows), list(rows.values()))
        except OSError:
            if final and require_save:
                raise SearchError("SEARCH_CACHE_WRITE_FAILED") from None
            if warn:
                warn(SAVE_WARNING)

    def encode(model, batch, width):
        return checked_vectors(np, model.encode(batch, batch_size=32, normalize_embeddings=True,
                                                show_progress_bar=False, convert_to_numpy=True), len(batch), width)

    todo, queued = [], set()
    for i, k in enumerate(keys):
        if k not in known and k not in queued:
            queued.add(k)
            todo.append(i)
    if keys:
        model = get_model()
        cached_width = len(next(iter(known.values()))) if known else None
        width, rest = model_dimension(model), todo
        if width is None:
            probe = todo[0] if todo else 0
            vector = encode(model, [texts[probe]], None)[0]
            width = vector.shape[0]
            if todo:
                known[keys[probe]] = vector
                rest = todo[1:]
        if cached_width is not None and cached_width != width:
            raise SearchError("SEARCH_FAILED")  # same model name, other vector width
        if todo and log:
            log(f"embedding {len(todo)} new inputs")
        for start in range(0, len(rest), step):
            part = rest[start:start + step]
            for i, vector in zip(part, encode(model, [texts[i] for i in part], width)):
                known[keys[i]] = vector
            if start + step < len(rest):
                save()
                if log:
                    log(f"  {start + len(part)}/{len(rest)} embedded, progress saved")
    keep = list(dict.fromkeys(keys)) if prune else list(dict.fromkeys(list(known) + keys))
    if todo or state == DISCARD or set(keep) != cached:
        save(final=True)
    if not keys:
        return np.zeros((0, 0), dtype="float32"), 0
    return np.stack([known[k] for k in keys]).astype("float32"), len(todo)
