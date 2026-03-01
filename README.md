# Polars List Utils (`polist`)

[`polist`](https://github.com/dashdeckers/polars_list_utils) is a Python package that provides a set of utilities for working with List-type columns in Polars DataFrames, especially for signal processing and feature extraction.

So far these utilities comprise those that I found to be missing or lacking from
the List namespace within the Polars library while I was working on a project at
work that required extensive handling of signal data which I was storing in Polars
DataFrames.

By providing these utilities as a Polars plugin, and thus not having to leave
the Polars DataFrame for these operations, I was able to significantly speed up
the processing of my data by benefiting from Polars query optimization and
parallelization. So while the operations themselves are not necessarily faster
than their Numpy counterparts (although they might be in some cases), the
integration with Polars gave my larger processing pipeline a significant speed
boost.

Status: Work-in-Progress!

## Features

- DSP and interpolation
        - `polist.apply_fft`
            - Applies FFT to list columns.
            - Supports windowing and Butterworth pre-filtering.
            - Supports normalization options.
        - `polist.interpolate_columns`
            - Interpolates `y_interp` from `x_data`, `y_data`, `x_interp`.
        - `polist.fft_freqs`, `polist.fft_freqs_linspace`
            - Helper functions for FFT/frequency axes and linspace generation.

- Missing-data handling (new)
        - `polist.fill_missing_list`
            - Forward/backward fill per list row.
            - Optional `limit` for maximum consecutive fills.
            - Treats nulls and `NaN` as missing.
        - `polist.interpolate_missing_list`
            - Interpolates missing values with `linear`, `nearest`, or `first_last` modes.
            - Treats nulls and `NaN` as missing.
        - `polist.missing_gap_flags`
            - Returns list-wise boolean flags for missing runs.
            - Supports minimum gap length via `min_gap`.

- List arithmetic and sorting
        - `polist.operate_scalar_on_list`
            - Applies `add`, `sub`, `mul`, `div` between list values and a scalar column.
            - Useful where Polars list eval with named columns is limited ([issue][list_eval_named]).
        - `polist.arg_sort_list`
            - Per-row `argsort` indices for list values.

- List joins and reshaping
        - `polist.inner_join_lists`, `polist.left_join_lists`, `polist.outer_join_lists`
            - Return per-row index pairs for joining list values.
            - Supports `join_nulls` behavior.
        - `polist.asof_join_lists`
            - Asof-style list joins with `backward`, `forward`, and `nearest` strategies.
            - Optional tolerance.
        - `polist.list_zip`, `polist.list_unzip`
            - Convert between two lists and list-of-struct / struct-of-lists forms.

- Aggregation and feature extraction
        - `polist.aggregate_list_col_elementwise`
            - Group-wise elementwise aggregations over list columns.
            - Supports `mean`, `sum`, `count`, `product`, `gmean`.
        - `polist.agg_of_range`, `polist.mean_of_range`
            - Aggregations over value ranges defined by a companion x-axis list.
            - Useful for timeseries/spectral feature extraction.

[list_eval_named]: https://github.com/pola-rs/polars/issues/7210
[elementwise_agg]: https://stackoverflow.com/questions/73776179/element-wise-aggregation-of-a-column-of-type-listf64-in-polars
[stack_overflow]: https://github.com/pola-rs/polars/issues/5455


### Example: (signal) -- (hann window) -- (FFT) -- (Freq. Normalization)

![DSP Example](examples/showcase_dsp.png)


## Installation (user)

```bash
uv pip install polars-list-utils
```

## Installation (developer)

1) Setup Rust (i.e. install rustup)
2) Setup Python (i.e. install uv)
3) Setup environment and compile plugin:

```bash
uv sync --extra dev
uv run maturin develop --release
```

4) (Maybe) configure Cargo to find uv's Python installs. For example:

```
# .cargo/config.toml
[env]
PYO3_PYTHON = "C:\\Users\\travis.hammond\\AppData\\Roaming\\uv\\python\\cpython-3.12.0-windows-x86_64-none\\python.exe"
```

5) Run:

```bash
uv run ./examples/showcase_dsp.py
```

6) Lint

```bash
uv run ruff check .
uv run ty check
cargo fmt
```

## Todo

- Add more features
- Add more tests