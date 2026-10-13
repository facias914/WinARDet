import glob
from distutils.core import setup, Extension

sources = glob.glob("fastevaluate_dota/*.cpp") + glob.glob("fastevaluate_dota/*.c")

module = Extension(
    "fastevaluate_dota._fastevaluate_dota",
    language="c++",
    sources=sources,
    extra_compile_args=["-O3"],
    extra_link_args=["-lpthread"],
)

setup(name='fastevaluate-dota',
    version='2.0',
    description='Fast COCO-style evaluation with DOTA maxDets',
    packages=['fastevaluate_dota'],
    ext_modules=[module],
    scripts=['scripts/cocoap_dota'],
)
