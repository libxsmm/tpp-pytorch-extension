import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if not os.environ.get("CC", None):
    os.environ["CC"] = "mpicc"

if not os.environ.get("CXX", None):
    os.environ["CXX"] = "mpicxx"

root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

ext_modules = [
    CppExtension(
        name="torch_mpi",
        sources=["src/init.cpp", "src/c10d/ProcessGroupMPI.cpp"],
        extra_compile_args=[
            "-Wformat",
            "-Wformat-security",
            "-D_FORTIFY_SOURCE=2",
            "-fstack-protector",
            "-DUSE_C10D_MPI",
            "-mavx512f",
            "-mavx512bw",
            "-mavx512vl",
        ],
        extra_link_args=["-Wl,-z,noexecstack", "-Wl,-z,relro", "-Wl,-z,now"],
        # include_dirs=[os.path.join(root_dir, "src")],
    )
]

setup(
    name="torch-mpi",
    version="2.3",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
