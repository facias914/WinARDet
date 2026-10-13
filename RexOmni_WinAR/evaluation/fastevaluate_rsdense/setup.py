import glob
from distutils.core import setup, Extension

sources = glob.glob("fastevaluate_rsdense/*.cpp") + glob.glob(
    "fastevaluate_rsdense/*.c"
)

module = Extension(
    "fastevaluate_rsdense._fastevaluate_rsdense",
    language="c++",
    sources=sources,
    extra_compile_args=["-O3"],
    extra_link_args=["-lpthread"],
)

setup(name='fastevaluate-rsdense',
    version='2.0',
    description='Uncapped COCO-style evaluation with global TP/FP/FN aggregation',
    packages=['fastevaluate_rsdense'],
    ext_modules=[module],
    scripts=['scripts/cocoap_rsdense'],
)
