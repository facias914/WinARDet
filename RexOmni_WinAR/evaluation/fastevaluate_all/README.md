# fastevaluate_all

This is a separate build of `fastevaluate_dota` that does not truncate COCO
predictions at `maxDets=1000`. Its effective `maxDets` is `INT_MAX`, so every
prediction loaded for each image/category participates in matching and metric
accumulation.

Build in place:

```bash
python setup.py build_ext --inplace
```

Python API:

```python
import fastevaluate_all as fe

result = fe.evaluate(gt_path, prediction_path, 0, 0, "coco")
```

The input formats and return dictionary are identical to `fastevaluate_dota`.
Filtering performed before evaluation (for example conversion with
`--positive_only`) is outside this package and must be disabled separately if
predictions from negative categories should also be retained.
