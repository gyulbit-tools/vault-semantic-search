"""Local semantic search for a folder of Markdown notes (an Obsidian-style vault).

Modules:
  config     configuration file and defaults
  markdown   front matter, section chunks and embedding windows
  pages      safe folder walk, exclusions, scopes and page reading
  embedding  local ONNX encoder and the embedding cache
  search     exact-term plus semantic retrieval, output and cache warming
  recall     recall@k measurement with your own question set
"""

__version__ = "0.1.0"
