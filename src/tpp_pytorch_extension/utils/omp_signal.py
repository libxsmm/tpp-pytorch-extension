import signal
import ctypes
import ctypes.util
import os

# On GNU/Linux the result is "libc.so.6"
libc_filename = ctypes.util.find_library("c")
libc = ctypes.cdll.LoadLibrary(libc_filename)

# declare libc signal()'s arguments and return types
# incorrect type declarations may create unpredictable results
libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
libc.signal.restype = ctypes.c_void_p

_debug_enabled = False


def omp_signal_handler():
    sighandler_original = libc.signal(signal.SIGUSR1, signal.SIG_IGN)
    # print("SigHan:", sighandler_original)
    if sighandler_original is not None:
        libc.signal(signal.SIGUSR1, sighandler_original)
        return True
    else:
        return False


def omp_debug_signal():
    if _debug_enabled:
        os.kill(os.getpid(), signal.SIGUSR1)


def enable_debug():
    global _debug_enabled
    omp_signal_handler()
    _debug_enabled = True


if __name__ == "__main__":
    import torch

    S = 4096
    x = torch.rand([S, S])
    # x = x + x
    x = x.mm(x)

    omp_debug_signal()
    x = x + x
    omp_debug_signal()
