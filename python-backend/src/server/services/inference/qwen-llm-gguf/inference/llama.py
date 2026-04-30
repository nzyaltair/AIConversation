import sys
import os
import ctypes
import time
from collections import deque, Counter
import numpy as np
from typing import List, Union, Set, Optional
from pathlib import Path

from .logger import info, warning, error, debug

# =========================================================================
# Configuration
# =========================================================================
LOGS = True

# =========================================================================
# Type Definitions
# =========================================================================

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),
        ("tensor_buft_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(ctypes.c_void_p)),
        ("n_samplers", ctypes.c_size_t),
    ]

class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]

class llama_logit_bias(ctypes.Structure):
    _fields_ = [
        ("token", llama_token),
        ("bias", ctypes.c_float),
    ]

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

# =========================================================================
# Llama.cpp Library Bindings
# =========================================================================

llama = None
ggml = None
ggml_base = None

llama_log_set = None
llama_backend_init = None
llama_backend_free = None
llama_model_default_params = None
llama_model_load_from_file = None
llama_model_free = None
llama_model_get_vocab = None
llama_context_default_params = None
llama_init_from_model = None
llama_free = None
llama_batch_init = None
llama_batch_free = None
llama_decode = None
llama_get_logits = None
llama_get_logits_ith = None
llama_get_embeddings = None
llama_tokenize = None
llama_vocab_n_tokens = None
llama_vocab_eos = None
llama_vocab_bos = None
llama_token_to_piece = None
llama_get_memory = None
llama_memory_clear = None
llama_model_n_embd = None

# Sampler
llama_sampler_chain_default_params = None
llama_sampler_chain_init = None
llama_sampler_chain_add = None
llama_sampler_init_greedy = None
llama_sampler_init_dist = None
llama_sampler_init_temp = None
llama_sampler_init_top_k = None
llama_sampler_init_top_p = None
llama_sampler_sample = None
llama_sampler_free = None
llama_sampler_init_min_p = None
llama_sampler_init_penalties = None
llama_sampler_accept = None


# llama.cpp 日志噪音模式：高度重复的逐条详情，跳过以减少日志量
_SKIP_PATTERNS = [
    " - kv ",
    "create_tensor: loading tensor",
    "control token:",
    "load:   - ",
]

def _is_noisy(msg: str) -> bool:
    if "assigned to device" in msg and "layer" in msg:
        return True
    if "llama_kv_cache: layer" in msg:
        return True
    for pat in _SKIP_PATTERNS:
        if pat in msg:
            return True
    return False


def logger_callback(level, message, user_data):
    if not message:
        return
    try:
        msg_str = message.decode('utf-8', errors='replace').strip()
        if not msg_str or msg_str in ['.', '\n']:
            return
        if _is_noisy(msg_str):
            return
        if level == 0:
            error(f"[llama.cpp] {msg_str}")
        elif level == 1:
            warning(f"[llama.cpp] {msg_str}")
        elif level == 2:
            info(f"[llama.cpp] {msg_str}")
        elif level >= 3:
            debug(f"[llama.cpp] {msg_str}")
        else:
            info(f"[llama.cpp] {msg_str}")
    except Exception as e:
        print(f"Log callback error: {e}")


def configure_logging(logs=True):
    global _log_callback_ref
    LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    if logs:
        _log_callback_ref = LOG_CALLBACK(logger_callback)
    else:
        _log_callback_ref = LOG_CALLBACK(lambda l, m, u: None)
    llama_log_set(_log_callback_ref, None)


def bind_llama_lib():
    global llama, ggml, ggml_base
    global llama_log_set, llama_backend_init, llama_backend_free
    global llama_model_default_params, llama_model_load_from_file, llama_model_free, llama_model_get_vocab
    global llama_context_default_params, llama_init_from_model, llama_free
    global llama_batch_init, llama_batch_free, llama_batch_get_one
    global llama_decode, llama_get_logits, llama_get_logits_ith, llama_get_embeddings, llama_tokenize
    global llama_get_memory, llama_memory_clear, llama_model_n_embd
    global llama_vocab_n_tokens, llama_vocab_eos, llama_vocab_bos, llama_token_to_piece
    global llama_sampler_chain_default_params, llama_sampler_chain_init, llama_sampler_chain_add
    global llama_sampler_init_greedy, llama_sampler_init_dist, llama_sampler_init_temp
    global llama_sampler_init_top_k, llama_sampler_init_top_p, llama_sampler_sample, llama_sampler_free
    global llama_sampler_init_min_p, llama_sampler_init_penalties, llama_sampler_accept
    global llama_sampler_init_logit_bias

    if llama is not None:
        return

    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")

    if sys.platform == "win32":
        GGML_DLL = "ggml.dll"
        GGML_BASE_DLL = "ggml-base.dll"
        LLAMA_DLL = "llama.dll"
    elif sys.platform == "darwin":
        GGML_DLL = "libggml.dylib"
        GGML_BASE_DLL = "libggml-base.dylib"
        LLAMA_DLL = "libllama.dylib"
    else:
        GGML_DLL = "libggml.so"
        GGML_BASE_DLL = "libggml-base.so"
        LLAMA_DLL = "libllama.so"

    ggml = ctypes.CDLL(os.path.join(lib_dir, GGML_DLL))
    ggml_base = ctypes.CDLL(os.path.join(lib_dir, GGML_BASE_DLL))
    llama = ctypes.CDLL(os.path.join(lib_dir, LLAMA_DLL))

    # Log callback
    LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    llama_log_set = llama.llama_log_set
    llama_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
    llama_log_set.restype = None
    configure_logging(logs=LOGS)

    # Load all backends (Vulkan, CUDA, etc.)
    ggml_backend_load_all = ggml.ggml_backend_load_all
    ggml_backend_load_all.argtypes = []
    ggml_backend_load_all.restype = None
    ggml_backend_load_all()

    llama_backend_init = llama.llama_backend_init
    llama_backend_init.argtypes = []
    llama_backend_init.restype = None
    llama_backend_init()

    # Backend
    llama_backend_free = llama.llama_backend_free
    llama_backend_free.argtypes = []
    llama_backend_free.restype = None

    # Model
    llama_model_default_params = llama.llama_model_default_params
    llama_model_default_params.argtypes = []
    llama_model_default_params.restype = llama_model_params

    llama_model_load_from_file = llama.llama_model_load_from_file
    llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    llama_model_load_from_file.restype = ctypes.c_void_p

    llama_model_free = llama.llama_model_free
    llama_model_free.argtypes = [ctypes.c_void_p]
    llama_model_free.restype = None

    llama_model_get_vocab = llama.llama_model_get_vocab
    llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    llama_model_get_vocab.restype = ctypes.c_void_p

    llama_model_n_embd = llama.llama_model_n_embd
    llama_model_n_embd.argtypes = [ctypes.c_void_p]
    llama_model_n_embd.restype = ctypes.c_int32

    # Context
    llama_context_default_params = llama.llama_context_default_params
    llama_context_default_params.argtypes = []
    llama_context_default_params.restype = llama_context_params

    llama_init_from_model = llama.llama_init_from_model
    llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    llama_init_from_model.restype = ctypes.c_void_p

    llama_free = llama.llama_free
    llama_free.argtypes = [ctypes.c_void_p]
    llama_free.restype = None

    # Batch
    llama_batch_init = llama.llama_batch_init
    llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama_batch_init.restype = llama_batch

    llama_batch_free = llama.llama_batch_free
    llama_batch_free.argtypes = [llama_batch]
    llama_batch_free.restype = None

    llama_batch_get_one = llama.llama_batch_get_one
    llama_batch_get_one.argtypes = [ctypes.POINTER(llama_token), ctypes.c_int32]
    llama_batch_get_one.restype = llama_batch

    # Decode
    llama_decode = llama.llama_decode
    llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    llama_decode.restype = ctypes.c_int32

    # Logits
    llama_get_logits = llama.llama_get_logits
    llama_get_logits.argtypes = [ctypes.c_void_p]
    llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    llama_get_logits_ith = llama.llama_get_logits_ith
    llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    llama_get_embeddings = llama.llama_get_embeddings
    llama_get_embeddings.argtypes = [ctypes.c_void_p]
    llama_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)

    # Tokenize
    llama_tokenize = llama.llama_tokenize
    llama_tokenize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(llama_token), ctypes.c_int32,
        ctypes.c_bool, ctypes.c_bool,
    ]
    llama_tokenize.restype = ctypes.c_int32

    # Vocab
    llama_vocab_n_tokens = llama.llama_vocab_n_tokens
    llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    llama_vocab_n_tokens.restype = ctypes.c_int32

    llama_vocab_eos = llama.llama_vocab_eos
    llama_vocab_eos.argtypes = [ctypes.c_void_p]
    llama_vocab_eos.restype = llama_token

    llama_vocab_bos = llama.llama_vocab_bos
    llama_vocab_bos.argtypes = [ctypes.c_void_p]
    llama_vocab_bos.restype = llama_token

    llama_token_to_piece = llama.llama_token_to_piece
    llama_token_to_piece.argtypes = [ctypes.c_void_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
    llama_token_to_piece.restype = ctypes.c_int

    # Memory (KV Cache)
    llama_get_memory = llama.llama_get_memory
    llama_get_memory.argtypes = [ctypes.c_void_p]
    llama_get_memory.restype = ctypes.c_void_p

    llama_memory_clear = llama.llama_memory_clear
    llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    llama_memory_clear.restype = None

    # Sampler
    llama_sampler_chain_default_params = llama.llama_sampler_chain_default_params
    llama_sampler_chain_default_params.argtypes = []
    llama_sampler_chain_default_params.restype = llama_sampler_chain_params

    llama_sampler_chain_init = llama.llama_sampler_chain_init
    llama_sampler_chain_init.argtypes = [llama_sampler_chain_params]
    llama_sampler_chain_init.restype = ctypes.c_void_p

    llama_sampler_chain_add = llama.llama_sampler_chain_add
    llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    llama_sampler_chain_add.restype = None

    llama_sampler_init_greedy = llama.llama_sampler_init_greedy
    llama_sampler_init_greedy.argtypes = []
    llama_sampler_init_greedy.restype = ctypes.c_void_p

    llama_sampler_init_dist = llama.llama_sampler_init_dist
    llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
    llama_sampler_init_dist.restype = ctypes.c_void_p

    llama_sampler_init_temp = llama.llama_sampler_init_temp
    llama_sampler_init_temp.argtypes = [ctypes.c_float]
    llama_sampler_init_temp.restype = ctypes.c_void_p

    llama_sampler_init_top_k = llama.llama_sampler_init_top_k
    llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    llama_sampler_init_top_k.restype = ctypes.c_void_p

    llama_sampler_init_top_p = llama.llama_sampler_init_top_p
    llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    llama_sampler_init_top_p.restype = ctypes.c_void_p

    llama_sampler_sample = llama.llama_sampler_sample
    llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    llama_sampler_sample.restype = llama_token

    llama_sampler_free = llama.llama_sampler_free
    llama_sampler_free.argtypes = [ctypes.c_void_p]
    llama_sampler_free.restype = None

    llama_sampler_init_logit_bias = llama.llama_sampler_init_logit_bias
    llama_sampler_init_logit_bias.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(llama_logit_bias)]
    llama_sampler_init_logit_bias.restype = ctypes.c_void_p

    llama_sampler_init_min_p = llama.llama_sampler_init_min_p
    llama_sampler_init_min_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    llama_sampler_init_min_p.restype = ctypes.c_void_p

    llama_sampler_init_penalties = llama.llama_sampler_init_penalties
    llama_sampler_init_penalties.argtypes = [ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    llama_sampler_init_penalties.restype = ctypes.c_void_p

    llama_sampler_accept = llama.llama_sampler_accept
    llama_sampler_accept.argtypes = [ctypes.c_void_p, llama_token]
    llama_sampler_accept.restype = None


def init():
    """
    Switch to DLL directory, initialize llama.cpp libraries, then restore CWD.
    """
    original_cwd = Path.cwd()
    lib_dir = Path(__file__).parent / 'bin'

    os.chdir(lib_dir)
    os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(os.getcwd())
    info(f"Initializing llama.cpp, cwd: {Path.cwd()}")

    bind_llama_lib()

    os.chdir(original_cwd)
    info(f"Restored cwd: {Path.cwd()}")

    return True


init()


# =========================================================================
# Llama.cpp High-Level API
# =========================================================================


class LlamaModel:
    """Model wrapper"""
    def __init__(self, path, n_gpu_layers=-1, use_gpu=True, backend="auto"):
        self.backend = backend
        self.ptr = self.load_model(path, n_gpu_layers=n_gpu_layers, use_gpu=use_gpu)

        self.vocab = llama_model_get_vocab(self.ptr)
        self.n_embd = llama_model_n_embd(self.ptr)
        self.eos_token = llama_vocab_eos(self.vocab)

    def load_model(self, model_path: str, n_gpu_layers: int = -1, use_gpu: bool = True):
        model_path = Path(model_path)

        model_params = llama_model_default_params()
        if use_gpu:
            model_params.n_gpu_layers = n_gpu_layers
        else:
            model_params.n_gpu_layers = 0
            model_params.devices = (ctypes.c_void_p * 1)(None)

        model = llama_model_load_from_file(
            model_path.as_posix().encode('utf-8'),
            model_params
        )

        if model:
            return model
        elif use_gpu:
            warning(f"GPU load failed, falling back to CPU: {model_path}")
            model_params.n_gpu_layers = 0
            model_params.devices = (ctypes.c_void_p * 1)(None)
            model = llama_model_load_from_file(
                model_path.as_posix().encode('utf-8'),
                model_params
            )
            if model:
                return model

        error(f"Model load failed: {model_path}")
        return None

    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = True) -> List[int]:
        """Text to token ID list"""
        return text_to_tokens(self.vocab, text, add_special, parse_special)

    def detokenize(self, tokens: List[int]) -> str:
        """Token ID list to text"""
        if tokens is None or len(tokens) == 0:
            return ""
        all_bytes = b"".join([self.token_to_bytes(tid) for tid in tokens])
        return all_bytes.decode('utf-8', errors='replace')

    def token_to_bytes(self, token_id: int) -> bytes:
        """Single token to bytes"""
        return token_to_bytes(self.vocab, token_id)

    def token_to_piece(self, token_id: int) -> str:
        """Single token to string piece"""
        return self.token_to_bytes(token_id).decode('utf-8', errors='replace')

    def token_bos(self) -> int:
        return llama_vocab_bos(self.vocab)

    def token_eos(self) -> int:
        return llama_vocab_eos(self.vocab)

    def token_to_id(self, text: str) -> int:
        """Single token string to ID (exact match only)"""
        res = self.tokenize(text, add_special=False, parse_special=True)
        return res[0] if res else -1

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            llama_model_free(self.ptr)
            self.ptr = None


class LlamaContext:
    """Context wrapper"""
    def __init__(self, model, n_ctx=2048, n_batch=2048, n_ubatch=512, n_seq_max=1,
                 embeddings=False, pooling_type=0, flash_attn=True,
                 offload_kqv=True, no_perf=True, n_threads=None, n_threads_batch=None):
        self.model = model
        params = llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_ubatch = n_ubatch
        params.n_seq_max = n_seq_max
        params.embeddings = embeddings
        params.pooling_type = pooling_type
        params.flash_attn_type = 1 if flash_attn else 0
        params.offload_kqv = offload_kqv
        params.no_perf = no_perf

        cpu_count = os.cpu_count() or 4
        if n_threads:
            params.n_threads = n_threads
        else:
            params.n_threads = cpu_count // 2

        if n_threads_batch:
            params.n_threads_batch = n_threads_batch
        else:
            params.n_threads_batch = n_threads if n_threads else cpu_count

        self.ptr = llama_init_from_model(model.ptr, params)
        if not self.ptr:
            raise RuntimeError("Context initialization failed")

    def decode(self, batch):
        struct = batch.struct if hasattr(batch, 'struct') else batch
        return llama_decode(self.ptr, struct)

    def decode_token(self, token_id):
        return self.decode(get_one_batch(token_id))

    def get_logits(self):
        return llama_get_logits(self.ptr)

    def get_logits_ith(self, i: int):
        return llama_get_logits_ith(self.ptr, i)

    def get_embeddings(self):
        return llama_get_embeddings(self.ptr)

    def clear_kv_cache(self):
        mem = llama_get_memory(self.ptr)
        llama_memory_clear(mem, True)

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            llama_free(self.ptr)
            self.ptr = None


class LlamaBatch:
    """Batch wrapper with direct property access"""
    def __init__(self, n_tokens, embd_dim=0, n_seq_max=1):
        self.struct = llama_batch_init(n_tokens, embd_dim, n_seq_max)
        self.n_tokens_max = n_tokens

    @property
    def n_tokens(self): return self.struct.n_tokens
    @n_tokens.setter
    def n_tokens(self, val): self.struct.n_tokens = val

    @property
    def token(self): return self.struct.token
    @property
    def embd(self): return self.struct.embd
    @property
    def pos(self): return self.struct.pos
    @property
    def n_seq_id(self): return self.struct.n_seq_id
    @property
    def seq_id(self): return self.struct.seq_id
    @property
    def logits(self): return self.struct.logits

    def set_embd(self, data: np.ndarray, pos: Union[np.ndarray, int] = 0, seq_id: int = 0):
        """High-level: inject embedding data and initialize positions (for ASR use)"""
        n_tokens = data.shape[0]
        if n_tokens > self.n_tokens_max:
            raise ValueError(f"Batch capacity exceeded: {n_tokens} > {self.n_tokens_max}")

        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        ctypes.memmove(self.embd, data.ctypes.data, data.nbytes)

        if isinstance(pos, int):
            pos_offset = pos
            for i in range(n_tokens):
                self.pos[i] = pos_offset + i
        elif isinstance(pos, np.ndarray):
            if not pos.flags['C_CONTIGUOUS']:
                pos = np.ascontiguousarray(pos)
            ctypes.memmove(self.pos, pos.ctypes.data, pos.nbytes)

        self.n_tokens = n_tokens
        for i in range(n_tokens):
            self.n_seq_id[i] = 1
            self.seq_id[i][0] = seq_id
            self.logits[i] = 1 if i == n_tokens - 1 else 0

        return self

    def __del__(self):
        if hasattr(self, 'struct'):
            llama_batch_free(self.struct)


def get_one_batch(token_id: int):
    """Zero-allocation single-token batch for autoregressive decoding"""
    token_arr = (llama_token * 1)(token_id)
    return llama_batch_get_one(token_arr, 1)


class LlamaSampler:
    """Sampler wrapper"""
    def __init__(
        self,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: Optional[int] = None,
        logit_bias: Optional[dict] = None,
        n_vocab: int = 0
    ):
        if seed is None:
            seed = int(time.time())

        sparams = llama_sampler_chain_default_params()
        self.ptr = llama_sampler_chain_init(sparams)

        # 1. Logit bias (highest priority)
        if logit_bias and n_vocab > 0 and isinstance(logit_bias, dict):
            n_bias = len(logit_bias)
            BiasArray = llama_logit_bias * n_bias
            bias_data = BiasArray()
            for i, (token, bias) in enumerate(logit_bias.items()):
                bias_data[i].token = token
                bias_data[i].bias = bias
            llama_sampler_chain_add(self.ptr, llama_sampler_init_logit_bias(n_vocab, n_bias, bias_data))

        # 2. Penalties
        has_penalty = (repeat_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0)
        if has_penalty:
            llama_sampler_chain_add(self.ptr, llama_sampler_init_penalties(
                penalty_last_n, repeat_penalty, frequency_penalty, presence_penalty
            ))

        # 3. Filters (order matters)
        if temperature > 0:
            if top_k > 0:
                llama_sampler_chain_add(self.ptr, llama_sampler_init_top_k(top_k))
            if top_p < 1.0:
                llama_sampler_chain_add(self.ptr, llama_sampler_init_top_p(top_p, 1))
            if 0.0 < min_p < 1.0:
                llama_sampler_chain_add(self.ptr, llama_sampler_init_min_p(min_p, 1))

            llama_sampler_chain_add(self.ptr, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(self.ptr, llama_sampler_init_dist(seed))
        else:
            llama_sampler_chain_add(self.ptr, llama_sampler_init_greedy())

        self._neg_inf = -1e10

    def accept(self, token_id: int):
        if self.ptr:
            llama_sampler_accept(self.ptr, token_id)

    def sample(self, ctx, idx=-1, limit_start=None, limit_end=None, allow_tokens=None):
        ctx_ptr = ctx.ptr if hasattr(ctx, 'ptr') else ctx

        if (limit_start is not None or limit_end is not None) and hasattr(ctx, 'get_logits'):
            n_vocab = llama_vocab_n_tokens(ctx.model.vocab)
            logits_ptr = ctx.get_logits_ith(idx)
            logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))

            s = max(0, limit_start) if limit_start is not None else 0
            e = min(n_vocab, limit_end) if limit_end is not None else n_vocab

            mask = np.ones(n_vocab, dtype=bool)
            mask[s:e] = False
            if allow_tokens:
                for t in allow_tokens:
                    if 0 <= t < n_vocab:
                        mask[t] = False

            logits[mask] = self._neg_inf

        return llama_sampler_sample(self.ptr, ctx_ptr, idx)

    def free(self):
        if hasattr(self, 'ptr') and self.ptr:
            llama_sampler_free(self.ptr)
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()

    def __del__(self):
        self.free()


# =========================================================================
# Utilities
# =========================================================================


def text_to_tokens(vocab, text, add_special=False, parse_special=True):
    text_bytes = text.encode("utf-8")
    n_tokens_max = len(text_bytes) + 32
    tokens = (llama_token * n_tokens_max)()
    n = llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_tokens_max, add_special, parse_special)
    return [tokens[i] for i in range(n)] if n >= 0 else []


def token_to_bytes(vocab, token_id):
    buf = ctypes.create_string_buffer(256)
    n = llama_token_to_piece(vocab, token_id, buf, ctypes.sizeof(buf), 0, True)
    return buf.raw[:n] if n > 0 else b""
