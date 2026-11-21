# vault-semantic-search

Local semantic search for a folder of Markdown notes, such as an Obsidian vault. Ask a question in your own words and get the notes whose sections are closest in meaning, with a small multilingual embedding model that runs on your CPU. Nothing leaves your machine and no model is downloaded at run time.

[한국어 README](README.ko.md)

## The problem

A keyword search (grep, the Obsidian search box) finds a note only when you remember the words it uses. Months later you remember what it was about, not how you wrote it: you ask "how long does the watering run before sunrise" and the note says "the drip line runs for twenty minutes every morning". An agent that answers questions from your notes has the same problem, and it tends to give up after a few grep misses and say there is nothing.

vault-semantic-search adds a meaning-based search next to grep:

1. Each note is cut into heading sections, and each section into overlapping windows that fit the model's 512-token input.
2. Every window is embedded once with `intfloat/multilingual-e5-small` (ONNX, CPU) and cached outside the vault. Later runs embed only new or changed text.
3. A question is embedded the same way and compared with every window. The best window per note decides the note's rank.
4. Exact matches of a question word in a note title, an alias or a chosen front matter field are put first (at most half of the slots), so a name you do remember is not lost behind looser semantic matches.
5. A `warm` command fills the cache in a nightly scheduled run, so a daytime search does not wait for embedding.

## What it does and does not do

- It reads only `*.md` files under the vault folder (or under the `--in` folders), and never writes notes. It writes only its own embedding cache and, when you ask `recall` for one, a result file (inside the vault only as a new file that is not a note).
- It never follows a link, junction or other reparse point inside the vault. Cloud placeholder files (OneDrive "files on demand") are read normally.
- It can skip folders entirely (`exclude_dirs`), and it can keep a page searchable while hiding its snippet and section title (`private_dirs`, `hide_snippets_when`). Excluded folders are skipped before any file in them is opened.
- Results are candidates. The number of results is the `--top` limit, not a total, and a short or empty list is not evidence that the vault has nothing on the subject. Every output says this.
- Output text is English. Questions and notes can be in any language the model supports; it was tested with English and Korean.
- It is not a chat or question-answering tool and does not call any LLM. It is not a full-text index: use grep for exact strings.
- It does not watch the vault. New notes are found on the next search (which embeds them) or the next `warm` run.

## Requirements

- Python 3.10 or later is the target. Tested only on Python 3.14 on Windows 11 (ARM64).
- `numpy`, `onnxruntime`, `tokenizers`, `PyYAML` (front matter) and `markdown-it-py` (code blocks); see `requirements.txt`, tested only with the versions named there. No PyTorch.
- The ONNX export of [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (about 470 MB). The model has its own license; see its model card.

The code is meant to run on Windows, macOS and Linux: its one platform call, the cache lock, uses `msvcrt.locking` on Windows and `fcntl.flock` elsewhere (both in the standard library). It has not been run on macOS or Linux.

## Install

```sh
git clone https://github.com/gyulbit-tools/vault-semantic-search
cd vault-semantic-search
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (cmd: .venv\Scripts\activate.bat)
# source .venv/bin/activate         # macOS, Linux
python -m pip install -r requirements.txt
```

Every `python` command below means the Python of this `.venv`: run it in the activated environment, or call `.venv\Scripts\python` (Windows) or `.venv/bin/python` (macOS, Linux) directly.

Download the model files once (the `hf` command comes with `huggingface_hub`; older versions call it `huggingface-cli download`):

```sh
python -m pip install -U huggingface_hub
hf download intfloat/multilingual-e5-small onnx/model.onnx onnx/tokenizer.json --local-dir models/multilingual-e5-small
```

The folder that holds `model.onnx` and `tokenizer.json` (`models/multilingual-e5-small/onnx` here) is your `model_dir`. The tool checks only that the two files exist. It does not check a hash; if you want a fixed model version, pass `--revision` to the download. This draft was tested with the files of revision `614241f6`.

## Configuration

A JSON file. The tool looks for it in this order: `--config FILE`, the `VAULT_SEARCH_CONFIG` environment variable, `.vault-search.json` in the vault root, then built-in defaults. `VAULT_SEARCH_MODEL_DIR` and `VAULT_SEARCH_CACHE_DIR` override `model_dir` and `cache_dir` from any of these. Every key is optional except that search needs `model_dir`. Unknown keys and wrong types stop the run with `CONFIG_INVALID` and the name of the key.

`.vault-search.json` is looked for only in the folder you pass as the vault. To search part of a vault, pass the vault root (the folder that holds `.vault-search.json`) together with `--in FOLDER`. If you pass a subfolder as the vault instead, name the original settings with `--config` or `VAULT_SEARCH_CONFIG`, or the folder rules of the root file do not apply.

A configuration file inside the vault is opened only when it is a regular file with one name (not a hard link) and no folder on its path is a link or junction; otherwise the run stops with `CONFIG_UNREADABLE` and a fixed reason, as it does when the file cannot be read. Whether a file is inside the vault is decided on its real path as well as on the path you gave, so a Windows short (8.3) name or other letter case gets the same rules (on a volume that ignores letter case, such as the macOS default, a folder spelled in other case is known to be the vault folder because it is the same folder on disk), and a path that reaches the vault through a link or junction (also one outside the vault, also a link to a file that does not exist yet) is refused the same way. Only a missing `.vault-search.json` means built-in defaults; an empty `--config` value stops the run with `CONFIG_UNREADABLE` (`detail=empty configuration path`) instead of falling back to it. The vault-root file is part of the vault: for a vault someone else gave you, run with your own `--config` file kept outside the vault, so that file is not used.

```json
{
  "model_dir": "C:/models/multilingual-e5-small/onnx",
  "model_name": "intfloat/multilingual-e5-small",
  "exclude_dirs": ["Archive", "Templates", "/Attachments", "Backups/*"],
  "private_dirs": ["Private"],
  "hide_snippets_when": [
    { "field": "private", "values": ["true"] },
    { "field": "sensitivity", "allow": ["low", "medium"] }
  ],
  "term_fields": ["aliases", "alias", "keywords"],
  "pin_field": "featured",
  "label_fields": ["type", "updated"],
  "require_scope_for_paths": false
}
```

| Key | Default | Meaning |
|---|---|---|
| `model_dir` | none | Folder with `model.onnx` and `tokenizer.json`. |
| `model_name` | `intfloat/multilingual-e5-small` | Name stored in the cache. Change it when you change the model files, or old vectors are reused. |
| `query_prefix`, `passage_prefix` | `"query: "`, `"passage: "` | Text put before questions and note windows. These are the E5 prefixes; use `""` for a model that has none. |
| `cache_dir` | `%LOCALAPPDATA%\vault-semantic-search`, else `$XDG_CACHE_HOME/vault-semantic-search`, else `~/.cache/vault-semantic-search` | Cache root. Each vault gets its own subfolder (`search.npz` and its lock file `search.npz.lock`). Keep it outside a synced vault. |
| `skip_dot_dirs` | `true` | Skip folders and files whose name starts with `.` (`.obsidian`, `.git`, ...). |
| `skip_dir_names` | `[".git", ".trash", "node_modules"]` | Folder names skipped at any depth (a note file of that name is not skipped). |
| `exclude_dirs` | `[]` | Folder patterns never read (see below). |
| `private_dirs` | `[]` | Folder patterns whose pages are searched and listed, but shown without snippet or section title, and whose front matter terms are not used. |
| `hide_snippets_when` | `[]` | Front matter rules with the same effect as `private_dirs`. `values`: hide when the field has one of the values. `allow`: hide unless the field has one of the values, so a page without the field is hidden too. Note values and rule values compare in one form: spaces trimmed, a value of several lines joined into one line with spaces, Unicode NFC normalisation and case folding. |
| `term_fields` | `["aliases", "alias", "keywords"]` | Front matter fields whose words count as exact terms, besides the note title. |
| `pin_field` | none | A front matter field; pages where it is `true` lead the exact-term matches. |
| `label_fields` | `[]` | Front matter fields shown after each result, such as a type or a date. |
| `require_scope_for_paths` | `false` | When `true`, a search without `--in` prints only the JSON summary, and `--output paths` or `details` without `--in` is refused (`SEARCH_SCOPE_REQUIRED`). Useful when an agent should narrow the folder before it sees note paths. |

Folder patterns (`exclude_dirs`, `private_dirs`) are vault-relative folder paths with `/`. Each part may use `*` (any run of characters) and `?` (one character); every other character, `[` and `]` included, matches itself, so `[Old] Notes` matches a folder of that name. A pattern matches at any depth (`Archive` also matches `Projects/Archive`); start it with `/` to match only at the vault root. Patterns are compared with folders only, never with a note's file name: `Backups/*` excludes (or makes private) the subfolders of `Backups` but not the notes directly in it. A link or junction inside the vault that leads to a folder and whose name an `exclude_dirs` pattern or a `skip_dir_names` name matches is skipped like an excluded folder, without being followed or counted as a read problem; a link to a file (or one whose target cannot be read) is a read problem whatever its name. A `private_dirs` pattern without `/` at the start is tested on the whole absolute path, so a page stays private even when you point the vault argument at the private folder itself. Names are compared after Unicode NFC normalisation and case folding.

## Usage

### search

```sh
python -I -B vault_search.py search <vault> "how long does the watering run before sunrise"
python -I -B vault_search.py search <vault> "question one" "question two" --in Projects --in "Journal/2026" --top 10
```

Several questions in one command share one folder scan and one model load. `--in` takes a vault-relative folder path (an absolute path must be inside the vault by its real spelling, or the run stops with `SEARCH_SCOPE_INVALID`). The folder at that path is used first. A single folder name that is not found at that path also matches one folder of that name anywhere in the vault (the output then says `scope_alias_resolved:`), and two such folders stop the run (`SEARCH_SCOPE_AMBIGUOUS`). If some folder cannot be read completely while the name is looked for (a link or junction that is not excluded, a folder that cannot be listed, a folder name that is not valid Unicode), the run stops with `SEARCH_SCOPE_UNRESOLVED`, because that folder could hold another folder of that name; exclude it or give the full path. A name never matches a folder under `private_dirs`; give its full path.

Output modes (`--output`):

| Mode | Shows |
|---|---|
| `details` (default) | Links, section titles and a 200-character snippet. |
| `paths` | Links with the similarity score and the `label_fields` values, without section titles or snippets (labels show for hidden pages too, except pages the `hide_snippets_when` rules could not check; see Limits). `--no-snippet` does the same in `details`. |
| `summary` | One JSON line per question with counts only, no paths. |
| `auto` | `details`, or `summary` for an unscoped search when `require_scope_for_paths` is on. |

Example: real output of the `intfloat/multilingual-e5-small` ONNX model on the synthetic test vault from `tests/support.py`, with `exclude_dirs: ["Archive"]`, `private_dirs: ["Private"]` and `label_fields: ["type"]`:

```sh
python -I -B vault_search.py search <vault> "how long does the watering run before sunrise" --top 3
```

```text
query: how long does the watering run before sunrise
SEARCH_COMPLETE: coverage_complete=true, candidates=3, chunks=8, read_problems=0
Candidates are exact-term matches plus the top semantic matches. The count is a limit, not a total, and is not evidence of absence.
1. [[Projects/Garden irrigation]] › Timer (similarity 0.89, type: project)
   The drip line runs for twenty minutes every morning before sunrise.
2. [[Notes/Shown by field]] › Plan (similarity 0.84)
   A watering can for the balcony plants.
3. [[Notes/Hidden by field]] › Plan (similarity 0.84)
   The drip watering schedule for the greenhouse.
```

A page under `Private` that ranked in the list would show `(snippet hidden: private folder or front matter rule)` in place of its snippet and no section title.

`SEARCH_PARTIAL` (`coverage_complete=false`) means some files or folders could not be read (an unreadable file, a link inside the vault, a note file with two names, text that is not valid UTF-8 or UTF-16). The results are still printed, but they do not cover the whole scope.

### warm

```sh
python -I -B vault_search.py warm <vault>
```

Embeds every section of the whole vault into the cache and drops vectors no longer used, then prints one JSON line (`WARM_COMPLETE` or `WARM_PARTIAL`, with counts). Progress goes to stderr. Only a complete scan of the whole vault drops vectors: `warm` or an unscoped search with no read problem, including one that finds no note or no section at all (that one needs no model, and a vault that never had a cache gets none). A scoped search, or a scan with a read problem, keeps the vectors of notes it did not read. If `warm` cannot save or clean the cache, or cannot name its folder (no home folder found), it stops with `SEARCH_CACHE_WRITE_FAILED` and exit code 2, also when it has nothing to save; a search that cannot save the cache, or name its folder, prints a warning on stderr and still answers from the vectors in memory.

The first run embeds everything and can take a long time on a large vault (tens of minutes). It saves its progress every 5,000 inputs, so an interrupted run continues where it stopped. Run it every night so the cache is warm when you search.

Windows Task Scheduler (PowerShell; `pythonw.exe` runs without a console window):

```powershell
$action = New-ScheduledTaskAction -Execute "C:\tools\vault-semantic-search\.venv\Scripts\pythonw.exe" `
  -Argument '-I -B -X utf8 "C:\tools\vault-semantic-search\vault_search.py" warm "C:\Notes\MyVault" --config "C:\tools\vault-semantic-search\config.json"'
$trigger = New-ScheduledTaskTrigger -Daily -At 3:30am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName "Vault Search Warm" -Action $action -Trigger $trigger -Settings $settings
```

`-StartWhenAvailable` runs a missed start after the computer wakes up. With `pythonw.exe` the output is discarded; the task's "Last Run Result" is the exit code (0 done, 2 blocked).

cron (macOS, Linux). Create the log folder once before you add the line, or the shell cannot open the log and the job does not run:

```sh
mkdir -p "$HOME/.cache/vault-semantic-search"
```

```cron
30 3 * * * /opt/vault-semantic-search/.venv/bin/python -I -B /opt/vault-semantic-search/vault_search.py warm "$HOME/Notes/MyVault" >> "$HOME/.cache/vault-semantic-search/warm.log" 2>&1
```

### recall

Measure how well search finds the notes you expect, on your own vault:

```sh
python -I -B vault_search.py recall <vault> --questions my-questions.json [--json-out result.json] [--top 10]
```

The question file (a synthetic example is in `examples/recall-questions.example.json`):

```json
{
  "scopes": ["Projects", "Journal", "Recipes"],
  "questions": [
    { "id": "q01", "query": "how long does the watering run before the sun comes up",
      "gold": ["Projects/Garden irrigation.md"] },
    { "id": "q02", "query": "stopping power got weak on my bike",
      "gold": ["Projects/Bicycle repair.md", "Journal/Bike shop visit.md"] }
  ]
}
```

`scopes` is optional (the default is the whole vault). `gold` lists one or more vault-relative note paths; a question counts as a hit at k when any gold note is in the top k. Write the questions the way you would ask them months later, not with the note's own words, or you measure grep.

A gold path uses `/` (or `\`) between folders, may leave out `.md`, and may not start with `/` or a drive or contain `.` or `..` parts, or on Windows contain `:` (`RECALL_GOLD_INVALID`). It names the note with exactly that spelling; if no note has it, the one note whose path is the same after Unicode NFC normalisation. Two such notes stop the run (`RECALL_GOLD_AMBIGUOUS`) instead of scoring either. A gold path with `.md` is compared with the whole file name, so `Note.md` and `Note.MD` are two notes; without `.md` it is compared with the path without the extension, so `Note` names both and stops the run. The result file's `gold_ranks` names each gold note with its own extension.

With `--json-out`, the result file is written before the numbers are printed; if it cannot be written, `recall` prints `RECALL_BLOCKED code=RECALL_OUTPUT_FAILED` and no numbers. An empty `--json-out` value is refused the same way, and so is a question file whose text cannot be written as UTF-8 (a lone surrogate in an id, a question, a gold path or a scope), before the model is loaded. The question file and the result file, when they are inside the vault, follow the same rules as a configuration file there. Inside the vault the result file must also be new and must not end in `.md` (`detail=existing file or note inside the vault`), and it is created only if no file of that name appeared during the run; outside the vault an existing file is replaced.

Report format (illustrative):

```text
HIT q01 rank=1 query: how long does the watering run before the sun comes up
HIT q02 rank=4 query: stopping power got weak on my bike
     gold Projects/Bicycle repair.md -> 4
     gold Journal/Bike shop visit.md -> not in top
RECALL_COMPLETE questions=2 top1=1/2 (50.0%) top5=2/2 (100.0%) top10=2/2 (100.0%)
```

The report shows only question ids, questions, gold notes and their ranks, never other result paths. A question set that cannot give a real denominator stops with exit code 2 before the model is loaded, instead of reporting 0%: an empty set, a question without gold notes, a gold note that search would not read (missing, outside `scopes`, excluded, empty), or a folder that could not be read completely.

## How well it finds things

Measured on one real personal vault on 2026-11-20 with the private tool this draft was extracted from (same chunking, model and ranking):

| | Hits | Recall |
|---|---|---|
| top 1 | 14 / 30 | 46.7% |
| top 5 | 21 / 30 | 70.0% |
| top 10 | 25 / 30 | 83.3% |

- 30 questions written by the vault's owner. Each asks about a note in words the note does not use (a question about the meaning, not a keyword lookup). Searches were limited with `--in` to the folders that hold the gold notes.
- Model: `intfloat/multilingual-e5-small`, ONNX, CPU. Notes and questions are mostly Korean, with English terms mixed in.
- The questions and the gold notes are personal and are not published. The numbers show what to expect, not a benchmark you can reproduce; run `recall` on your own vault to get yours.

The code in this repository was then run on the same vault with the same question set and settings that copy the private tool's folder rules: all 30 questions got the same rank as before (14, 21 and 25 hits).

Speed, measured on a Windows 11 ARM64 laptop with Python 3.14 and onnxruntime 1.32 (CPU), warm cache: about 1.7 seconds for one question with the private tool, and about 18 seconds for the 30 recall questions in one command with this code (one scan, one model load). A cold cache is slower by the time it takes to embed the new text.

## Limits

- **One small model.** e5-small is fast and multilingual but not the strongest model. Pooling is mean pooling with L2 normalisation, as in the E5 sentence-transformers export; a model that needs CLS pooling or other input names is not supported without code changes. Only `intfloat/multilingual-e5-small` was tested.
- **Long sections** are split into overlapping windows of up to 500 characters (100 characters overlap), each cut to fit 512 tokens with the title header. A note is ranked by its single best window.
- **Exact terms** are whole words after Unicode NFC and case folding: no stemming, no substrings. A word that appears in one fifth or more of the searched pages is ignored as too common.
- **Front matter** is read as plain YAML only: scalars, lists and mappings, with every value kept as the text you wrote (`true` and `2026-01-02` stay text). A tag (`!!int 10`, `!!str x`, `!custom x`), an anchor (`&a`), an alias (`*a`) or a merge key (`<<`, unquoted) anywhere in it makes the whole front matter unreadable. It is parsed with PyYAML's pure Python parser, also where PyYAML's C extension is installed, because the two accept different text; so every machine reads a note the same way. It starts with a `---` first line and ends at the next line that is `---` at the start of the line (spaces or tabs may follow); `...` does not end it. Lines end only at a line feed (`\n`, also in `\r\n`): a lone `\r`, U+0085, U+2028 or U+2029 does not end a line there. A front matter that holds a control character or a line or paragraph separator other than the line feed and the tab (a lone `\r`, U+0085, U+2028, U+2029, U+000B, U+000C and the like), or only white space, is unreadable too, because YAML would end lines where this tool does not; an empty one (`---` right after `---`) has no fields. Top-level text values and lists of text are used; a `>` or `|` block becomes one line. Nested mappings and a key given twice are not read. Without `hide_snippets_when` rules a page whose front matter is unreadable is still searched and shown, but none of its front matter is used (no labels, exact terms or `pin_field`).
- **Pages the rules cannot check.** When `hide_snippets_when` rules are set, a page is hidden and shown with its link only, without labels, when the rules cannot be checked on it: its file is not UTF-8 text (UTF-16 and text with NUL included), its first line is `---` with no closing `---` line, its first line is `---` only once other white space is removed (`---` followed by U+2028, U+2029, U+0085 or a no-break space, or a `---` after a space or tab; another reader may see no front matter there), its front matter is unreadable (not valid YAML, it uses a tag, anchor, alias or merge key, it holds a control character or line separator, or it holds only white space), or a rule field is nested or given twice. Its front matter is also not used for exact terms or `pin_field`. Search goes on with the other pages. Without rules such a page is searched and shown, but a front matter that is unreadable (see above) or whose first line is `---` only once other white space is removed is not used (no labels, exact terms or `pin_field`); the other cases are read as before.
- **Hiding is display only.** Hidden pages are still embedded into the cache, and their links are still shown, with their labels except for the pages above. The cache file holds vectors and SHA-1 keys, not note text, but treat it as derived from your notes.
- **Hard links.** A note file with more than one name is refused before it is opened and counted as a read problem (so `recall` stops), because one name could be outside a private or excluded folder and the tool cannot tell which name is the original. This works where the file system reports the link count; where it does not, such a file is read like any other. To search that note, keep one name.
- **Code blocks** (fenced and indented, by the CommonMark rules, also inside lists and quotes) are not searched; inline code is searched without the backticks. Embedded files (`![[...]]`) and image links are dropped; link text is kept.
- **Cache key** is the model name plus the exact input text of a note window. Changing the chunking or `passage_prefix` re-embeds only the windows whose final input text changes (a new `passage_prefix` changes all of them). Questions have no cache entries of their own: when there is at least one note window, a question is embedded on each run, except that a question whose final input (`query_prefix` plus the question) equals a note window's input uses that window's vector. When there are no chunks, only exact-term matches are returned and the model is not loaded. Changing `query_prefix` re-embeds no note window. A cache file is used only when its format number is the integer 2, every key has one finite unit-length vector and a stored SHA-256 digest of the model name, keys and vectors matches; any other file (a damaged one included) is rebuilt. A vector the model computes, for a note window or for a question, is held to the same rule: a vector that is not finite and of unit length (or a question vector of another width than the windows') stops the run with `SEARCH_FAILED` before it is saved or ranked (progress saved before it is kept). Runs that save the same cache take turns through its lock file, and each save keeps the vectors another run saved meanwhile; only a complete whole-vault run drops vectors. A lock not got within 30 seconds is a failed save. A valid cache whose vectors have another width than the model's (the same `model_name` with other model files) stops the run with `SEARCH_FAILED` before the cache is changed: change `model_name`. Each vault's cache folder is named after the vault path as the file system spells it (on macOS and Linux, each folder name as its parent folder lists it), so the same vault given in other letter case, for example on a volume that ignores case such as the macOS default, uses the same cache. If a parent folder's listing cannot be read, the path component immediately below it keeps the spelling given, so changing the letter case of that component makes a separate cache.
- **No index structure.** Every search compares the question with every cached vector in memory, which is fine for tens of thousands of sections but not for millions.
- **Local vault paths only.** A UNC path (`\\server\share\...`, `\\?\UNC\...`, also as an `--in` value) and a mapped network drive are refused (`SEARCH_ROOT_INVALID`), and so is a vault path in which the vault folder or any folder above it is a symlink or junction, so the private-folder and scope checks always see the real path. This includes system links such as macOS `/tmp` (a link to `/private/tmp`): give the real path instead.
- **Paths in results** are vault-relative links (`[[folder/note]]`). When two searched notes would get the same link (`Note.md` and `Note.MD`), their links keep the extension (`[[folder/Note.md]]`, `[[folder/Note.MD]]`), whichever of them are in the results. A note whose link would then equal one of those (`Note.md.md` next to those two, whose plain link is `[[folder/Note.md]]`) keeps its extension too. Links are compared with their exact spelling, so `Note.md` and `note.md` are `[[folder/Note]]` and `[[folder/note]]`. A blocked search prints one JSON line with a fixed code (and, for `CONFIG_INVALID`, the key name; for a refused configuration file in the vault, a fixed reason), never paths, questions or note text; invalid command-line arguments (a `--top` out of range included) print `SEARCH_ARGS_INVALID` the same way, without the arguments. Set `VAULT_SEARCH_DEBUG=1` to see a traceback on stderr. A blocked `recall` names the question ids and the gold paths from your own question file. A folder or note whose name cannot be written as UTF-8 (a broken Unicode name) is counted as a read problem and skipped, also a folder given with `--in` (once, also when another `--in` folder holds it or several `--in` values stop at the same broken folder); the other `--in` folders are still searched, and the search never widens to the whole vault.

## Tests

```sh
python -I -B tests/run_tests.py
```

Run them with the `.venv` Python from Install. The tests build small synthetic vaults in a temporary folder and use a fake encoder, so they need `numpy`, `PyYAML` and `markdown-it-py` but no model (a search with a loader other than the built-in ONNX one checks only for `numpy`). Set `VAULT_SEARCH_MODEL_DIR` to the model folder to also run the ONNX smoke tests (`tests/test_onnx_model.py`). The junction, short-name and extended-path tests run only on Windows, and the symlink tests skip when the account may not create symlinks. The last line is `RESULT pass=N fail=M skip=K`; any failure exits 1.

## License

MIT. See [LICENSE](LICENSE). The model is not part of this repository and has its own license.
