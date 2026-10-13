import glob
from distutils.core import Extension, setup


sources = glob.glob("fastevaluate_all/*.cpp") + glob.glob("fastevaluate_all/*.c")

module = Extension(
    "fastevaluate_all._fastevaluate_all",
    language="c++",
    sources=sources,
    extra_compile_args=["-O3"],
    extra_link_args=["-lpthread"],
)

setup(
    name="fastevaluate-all",
    version="1.0",
    description="Fast COCO/LVIS evaluation using every prediction per image",
    packages=["fastevaluate_all"],
    ext_modules=[module],
    scripts=["scripts/cocoap_all"],
)
