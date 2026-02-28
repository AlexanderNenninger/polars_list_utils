import math

import polars as pl
import polars_list_utils as polist
import pytest


def test_inner_join_lists_basic_and_null_rows() -> None:
    df = pl.DataFrame(
        {
            "x": [[10, 20, 30], [None, 1], None],
            "y": [[20, 30, 40], [None, 2], [1]],
        }
    )
    result_df = df.with_columns(
        polist.inner_join_lists("x", "y", join_nulls=False)
        .struct.unnest()
        .name.suffix("_indices"),
    ).with_columns(
        pl.col("x").list.gather(pl.col("x_indices")).name.suffix("_joined"),
        pl.col("y").list.gather(pl.col("y_indices")).name.suffix("_joined"),
    )

    assert result_df.select(
        (
            (pl.col("x_joined").list.set_difference("y").list.len() == 0)
            | pl.col("x_joined").is_null()
        ).all()
    ).item(), "All joined elements from x should be in y"

    assert result_df.select(
        (
            (pl.col("y_joined").list.set_difference("x").list.len() == 0)
            | pl.col("y_joined").is_null()
        ).all()
    ).item(), "All joined elements from y should be in x"


def test_argsort_list_basic() -> None:
    df = pl.DataFrame(
        {
            "x": [[3, 1, 2], [None, 2, 1], None],
        }
    )

    result = df.with_columns(
        x_argsort=polist.arg_sort_list("x", descending=False, nulls_last=True)
    ).with_columns(x_sorted=pl.col("x").list.gather(pl.col("x_argsort")))

    assert result.get_column("x_argsort").to_list() == [[1, 2, 0], [2, 1, 0], None]
    assert result.get_column("x_sorted").to_list() == [[1, 2, 3], [1, 2, None], None]


def test_asof_join_lists_strategies() -> None:
    df = pl.DataFrame(
        {
            "x": [[1.0, 2.5, 4.1, None], [5.0], None],
            "y": [[1.0, 2.0, 3.0, 5.0], [1.0, 6.0], [1.0]],
        }
    )

    backward = (
        df.with_columns(
            polist.asof_join_lists("x", "y", strategy="backward")
            .struct.unnest()
            .name.suffix("_idx"),
        )
        .select("x_idx", "y_idx")
        .to_dict(as_series=False)
    )
    assert backward["x_idx"] == [[0, 1, 2], [0], None]
    assert backward["y_idx"] == [[0, 1, 2], [0], None]

    forward = (
        df.with_columns(
            polist.asof_join_lists("x", "y", strategy="forward")
            .struct.unnest()
            .name.suffix("_idx"),
        )
        .select("x_idx", "y_idx")
        .to_dict(as_series=False)
    )
    assert forward["x_idx"] == [[0, 1, 2], [0], None]
    assert forward["y_idx"] == [[0, 2, 3], [1], None]

    nearest = (
        df.with_columns(
            polist.asof_join_lists("x", "y", strategy="nearest")
            .struct.unnest()
            .name.suffix("_idx"),
        )
        .select("x_idx", "y_idx")
        .to_dict(as_series=False)
    )
    assert nearest["x_idx"] == [[0, 1, 2], [0], None]
    assert nearest["y_idx"] == [[0, 1, 3], [1], None]


def test_operate_scalar_on_list_all_operations() -> None:
    df = pl.DataFrame(
        {
            "vals": [[1.0, 2.0, None], None],
            "s": [2.0, 3.0],
        }
    )

    out = df.with_columns(
        add=polist.operate_scalar_on_list("vals", "s", operation="add"),
        sub=polist.operate_scalar_on_list("vals", "s", operation="sub"),
        mul=polist.operate_scalar_on_list("vals", "s", operation="mul"),
        div=polist.operate_scalar_on_list("vals", "s", operation="div"),
    )

    assert out.get_column("add").to_list() == [[3.0, 4.0, None], None]
    assert out.get_column("sub").to_list() == [[-1.0, 0.0, None], None]
    assert out.get_column("mul").to_list() == [[2.0, 4.0, None], None]
    assert out.get_column("div").to_list() == [[0.5, 1.0, None], None]


def test_interpolate_columns_basic() -> None:
    df = pl.DataFrame(
        {
            "x_data": [[0.0, 1.0, 2.0], [0.0, 2.0], None],
            "y_data": [[0.0, 10.0, 20.0], [0.0, 20.0], [1.0]],
            "x_interp": [[0.5, 1.5, 3.0], [1.0, 3.0], [0.0]],
        }
    )

    out = df.with_columns(
        y_interp=polist.interpolate_columns("x_data", "y_data", "x_interp")
    )

    assert out.get_column("y_interp").to_list() == [
        [5.0, 15.0, 20.0],
        [10.0, 20.0],
        None,
    ]


def test_aggregate_list_col_elementwise_from_example_pattern() -> None:
    df = pl.DataFrame(
        {
            "list_col": [
                [None, 0.0, 0.0, 0.0],
                [float("nan"), 1.0, 0.0, 0.0],
                [float("nan"), float("nan"), float("nan"), float("nan")],
            ]
        }
    )

    out = df.group_by(pl.lit(1)).agg(
        mean=polist.aggregate_list_col_elementwise(
            "list_col", list_size=4, aggregation="mean"
        ),
        summ=polist.aggregate_list_col_elementwise(
            "list_col", list_size=4, aggregation="sum"
        ),
        count=polist.aggregate_list_col_elementwise(
            "list_col", list_size=4, aggregation="count"
        ),
        mean_short=polist.aggregate_list_col_elementwise(
            "list_col", list_size=2, aggregation="mean"
        ),
    )

    row = out.row(0, named=True)
    assert row["mean"] == [None, 0.5, 0.0, 0.0]
    assert row["summ"] == [None, 1.0, 0.0, 0.0]
    assert row["count"] == [0.0, 2.0, 2.0, 2.0]
    assert row["mean_short"] == [None, 0.5]


def test_mean_and_agg_of_range_from_example_pattern() -> None:
    df = pl.DataFrame(
        {
            "y_values": [
                [float("nan"), 1.0] + [0.0] * 10,
                [None, 0.0, 0.0],
                [float("nan")] * 10,
                [1.0, 8.0, 3.0, 2.0],
                [4.0, 5.0, 2.0, 3.0],
            ]
        }
    ).with_columns(pl.lit(list(range(10))).cast(pl.List(pl.Float64)).alias("x_axis"))

    out = df.with_columns(
        mean_of_range=polist.mean_of_range("y_values", "x_axis", x_min=0, x_max=1),
        mean_of_offset=polist.mean_of_range(
            "y_values", "x_axis", x_min=0, x_max=3, x_min_idx_offset=1
        ),
        median=polist.agg_of_range(
            "y_values", "x_axis", aggregation="median", x_min=0, x_max=3
        ),
        median_exclude_inner=polist.agg_of_range(
            "y_values",
            "x_axis",
            aggregation="median",
            x_min=0,
            x_max=3,
            x_range_excluded=(1.0, 2.0),
        ),
    )

    assert out.get_column("mean_of_range").to_list() == [1.0, 0.0, None, 4.5, 4.5]
    assert out.get_column("mean_of_offset").to_list() == [
        0.0,
        0.0,
        None,
        pytest.approx(13.0 / 3.0),
        pytest.approx(10.0 / 3.0),
    ]
    assert out.get_column("median").to_list() == [0.0, 0.0, None, 2.5, 3.5]
    assert out.get_column("median_exclude_inner").to_list() == [
        0.0,
        None,
        None,
        1.5,
        3.5,
    ]


def test_fft_helpers_and_apply_fft() -> None:
    assert polist.fft_freqs(n=3, fs=8) == [0.0, 2.0, 4.0]

    x_lin = polist.fft_freqs_linspace(fnum=5, fmax=10.0)
    assert len(x_lin) == 5
    assert x_lin[0] == 0.0
    assert x_lin[-1] == 10.0

    df = pl.DataFrame(
        {
            "s": [
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0, 0.0],
            ]
        }
    )

    out = df.with_columns(
        s_passthrough=polist.apply_fft("s", sample_rate=8, skip_fft=True),
        s_fft=polist.apply_fft("s", sample_rate=8, norm="length"),
    )

    assert out.get_column("s_passthrough").to_list() == out.get_column("s").to_list()

    fft_lists = out.get_column("s_fft").to_list()
    assert [len(v) for v in fft_lists] == [3, 3]
    assert all(math.isfinite(val) for row in fft_lists for val in row)
