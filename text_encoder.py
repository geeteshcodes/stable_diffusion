"""
Text Encoder for Text-to-Image Diffusion (v2: token-level conditioning).

encode() returns PACKED token embeddings of shape (N, TOKEN_LEN, 385):
  [..., :384] = MiniLM token embeddings (zeroed on padding), [..., 384] = 1.0 for real tokens, 0.0 for padding.
An all-zero tensor is the unconditional (null) prompt used for classifier-free guidance.
Runs on CPU, zero GPU VRAM.
"""

import zlib
import numpy as np

TEXT_EMBED_DIM = 384
TOKEN_LEN = 24


class TextEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2", embed_dim=TEXT_EMBED_DIM):
        self.embed_dim = embed_dim
        self.token_len = TOKEN_LEN
        self.backend = "fallback"
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[TextEncoder] Loading SentenceTransformer '{model_name}' on CPU...")
            self.model = SentenceTransformer(model_name, device="cpu")
            self.model.max_seq_length = TOKEN_LEN
            self.embed_dim = self.model.get_sentence_embedding_dimension()
            self.backend = "sentence_transformers"
            print(f"[TextEncoder] Ready: {TOKEN_LEN} tokens x {self.embed_dim} dims.")
        except ImportError:
            print("[TextEncoder] 'sentence-transformers' not installed. Run: pip install sentence-transformers")

    @property
    def cond_shape(self):
        return (self.token_len, self.embed_dim + 1)

    def encode(self, texts, fp16=False):
        """Packed token embeddings, shape (N, TOKEN_LEN, embed_dim + 1)."""
        if isinstance(texts, str):
            texts = [texts]
        if self.backend != "sentence_transformers":
            raise RuntimeError("Token-level conditioning requires sentence-transformers.")
        import torch
        feats = self.model.tokenize(list(texts))
        with torch.no_grad():
            out = self.model.forward(feats)
        tok = out["token_embeddings"].cpu().numpy().astype(np.float32)
        mask = out["attention_mask"].cpu().numpy().astype(np.float32)
        L = tok.shape[1]
        if L < TOKEN_LEN:
            tok = np.pad(tok, ((0, 0), (0, TOKEN_LEN - L), (0, 0)))
            mask = np.pad(mask, ((0, 0), (0, TOKEN_LEN - L)))
        else:
            tok, mask = tok[:, :TOKEN_LEN], mask[:, :TOKEN_LEN]
        tok = tok * mask[..., None]
        packed = np.concatenate([tok, mask[..., None]], axis=-1)
        return packed.astype(np.float16) if fp16 else packed

    def encode_pooled(self, texts):
        """Legacy sentence-level embedding (N, embed_dim); not used by the v2 model."""
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "sentence_transformers":
            return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
        out = []
        for text in texts:
            vec = np.zeros(self.embed_dim, dtype=np.float32)
            for word in text.lower().split():
                vec[zlib.crc32(word.encode("utf-8")) % self.embed_dim] += 1.0
            n = np.linalg.norm(vec)
            out.append(vec / n if n > 1e-6 else vec)
        return np.array(out, dtype=np.float32)

    def get_null_embedding(self, batch_size=1):
        return np.zeros((batch_size, self.token_len, self.embed_dim + 1), dtype=np.float32)
