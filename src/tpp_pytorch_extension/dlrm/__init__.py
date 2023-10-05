# from . import mlp
from contextlib import contextmanager


@contextmanager
def tpp_impl(use_TPP=True, use_bf16=False):
    with mlp.tpp_impl(use_TPP, use_bf16):
        yield
