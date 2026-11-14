# Changelog

## 0.1.1
- Lower peak memory: about 5x the size of M (was 7x) with block-coordinate
  steps and 7x (was 11x) with APG. Iterates are updated in place and
  elementwise work runs in cache-sized row blocks.
- About 10% faster per iteration on large matrices.
- Results unchanged (identical up to floating-point rounding).

## 0.1.0
- First release.
