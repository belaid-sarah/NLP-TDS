import os
import re

import numpy as np
import openai
import tiktoken
import yaml

from FlagEmbedding import FlagModel

# Lazy-load config so the module can be imported without config.yml
_CONF = None
_CLIENT = None

tokenizer = tiktoken.get_encoding("cl100k_base")


def _get_conf():
    global _CONF
    if _CONF is None:
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        # Support ${ENV_VAR} substitution in values
        for key, val in raw.items():
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                env_var = val[2:-1]
                raw[key] = os.environ.get(env_var, "")
        _CONF = raw
    return _CONF


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        conf = _get_conf()
        _CLIENT = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=conf["groq_key"],
        )
    return _CLIENT


def get_model(config):
    if config and config.get("model"):
        return RAG(**config["model"])
    return RAG()


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------

def parse_markdown_sections(md_text: str) -> list[dict]:
    """
    Parses markdown into sections preserving header hierarchy.
    Returns list of {'headers': [...], 'content': str}
    """
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack: list[str] = []
    current_section: dict = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            if current_section["content"]:
                sections.append(current_section)
            header_stack = header_stack[: level - 1]
            header_stack.append(title)
            current_section = {"headers": header_stack.copy(), "content": ""}
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return sections


def chunk_markdown(md_text: str, chunk_size: int = 128, overlap: int = 0) -> list[str]:
    """
    Split a markdown document into token chunks.

    Args:
        md_text:    Raw markdown text.
        chunk_size: Max number of tokens per chunk.
        overlap:    Number of tokens to overlap between consecutive chunks.
                    0 means no overlap (original behaviour).
    """
    sections = parse_markdown_sections(md_text)
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)

    for section in sections:
        tokens = tokenizer.encode(section["content"])
        for i in range(0, len(tokens), step):
            token_chunk = tokens[i : i + chunk_size]
            if token_chunk:
                chunks.append(tokenizer.decode(token_chunk))

    return chunks


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# RAG model
# ---------------------------------------------------------------------------

class RAG:
    """
    Retrieval-Augmented Generation model.

    Parameters
    ----------
    chunk_size      : tokens per chunk used for embedding / retrieval.
    overlap         : token overlap between consecutive chunks (0 = no overlap).
    use_small2big   : if True, retrieve using small chunks but answer with
                      the larger parent chunk for richer context.
    big_chunk_size  : size of the parent (big) chunks when use_small2big=True.
    llm_model       : Groq model used for generation.
    embedder_name   : HuggingFace model used for embeddings.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 0,
        use_small2big: bool = False,
        big_chunk_size: int = 512,
        llm_model: str = "llama-3.3-70b-versatile",
        embedder_name: str = "BAAI/bge-base-en-v1.5",
    ):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._use_small2big = use_small2big
        self._big_chunk_size = big_chunk_size
        self._llm_model = llm_model
        self._embedder_name = embedder_name

        self._embedder = None
        self._loaded_files: set = set()
        self._texts: list[str] = []

        # Chunks used for embedding (small if small2big, else normal)
        self._chunks: list[str] = []
        self._corpus_embedding: np.ndarray | None = None

        # Only populated when use_small2big=True
        self._big_chunks: list[str] = []
        self._small_to_big: list[int] = []  # small chunk idx → big chunk idx

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def load_files(self, filenames):
        new_texts = []
        for filename in filenames:
            fname = str(filename)
            if fname in self._loaded_files:
                continue
            with open(filename, encoding="utf-8") as f:
                text = f.read()
            new_texts.append(text)
            self._texts.append(text)
            self._loaded_files.add(fname)

        if not new_texts:
            return

        new_chunks = self._compute_chunks(new_texts)
        self._chunks += new_chunks

        new_embedding = self.embed_corpus(new_chunks)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _compute_chunks(self, texts: list[str]) -> list[str]:
        if self._use_small2big:
            return self._compute_small2big_chunks(texts)

        chunks: list[str] = []
        for txt in texts:
            chunks += chunk_markdown(txt, chunk_size=self._chunk_size, overlap=self._overlap)
        return chunks

    def _compute_small2big_chunks(self, texts: list[str]) -> list[str]:
        """
        Build two levels of chunks:
          - big chunks (parent) stored in self._big_chunks
          - small chunks (child) stored in self._chunks, used for retrieval
        Each small chunk maps to its parent big chunk index via self._small_to_big.
        """
        small_chunks: list[str] = []

        for txt in texts:
            big_for_text = chunk_markdown(txt, chunk_size=self._big_chunk_size, overlap=0)
            for big_chunk in big_for_text:
                big_idx = len(self._big_chunks)
                self._big_chunks.append(big_chunk)

                # Split the big chunk into small pieces
                tokens = tokenizer.encode(big_chunk)
                step = max(1, self._chunk_size - self._overlap)
                for i in range(0, len(tokens), step):
                    small_tokens = tokens[i : i + self._chunk_size]
                    if small_tokens:
                        small_chunks.append(tokenizer.decode(small_tokens))
                        self._small_to_big.append(big_idx)

        return small_chunks

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def get_embedder(self) -> FlagModel:
        if self._embedder is None:
            self._embedder = FlagModel(
                self._embedder_name,
                query_instruction_for_retrieval=(
                    "Represent this sentence for searching relevant passages:"
                ),
                use_fp16=True,
            )
        return self._embedder

    def embed_corpus(self, chunks: list[str]) -> np.ndarray:
        return self.get_embedder().encode(chunks)

    def embed_questions(self, questions: list[str]) -> np.ndarray:
        return self.get_embedder().encode(questions)

    def get_corpus_embedding(self) -> np.ndarray | None:
        return self._corpus_embedding

    def get_chunks(self) -> list[str]:
        return self._chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _get_context(self, query: str, top_k: int = 5) -> list[str]:
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T
        # argsort ascending → take last top_k (highest scores)
        top_indices = list(np.argsort(sim_scores[0]))[-top_k:]

        if self._use_small2big and self._big_chunks:
            # Map each small-chunk index to its parent big-chunk index,
            # de-duplicate while preserving relevance order (highest first).
            seen: set[int] = set()
            big_indices: list[int] = []
            for i in reversed(top_indices):
                big_idx = self._small_to_big[i]
                if big_idx not in seen:
                    seen.add(big_idx)
                    big_indices.append(big_idx)
            return [self._big_chunks[i] for i in big_indices[:top_k]]

        return [self._chunks[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def reply(self, query: str) -> str:
        prompt = self._build_prompt(query)
        res = _get_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._llm_model,
        )
        return res.choices[0].message.content

    def _build_prompt(self, query: str) -> str:
        context_str = "\n\n".join(self._get_context(query))
        return (
            "Context information is below.\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            'If the answer is not in the context information, reply "I cannot answer that question".\n'
            f"Query: {query}\n"
            "Answer:"
        )
