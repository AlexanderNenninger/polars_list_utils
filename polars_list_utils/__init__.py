from pathlib import Path
from typing import Optional, Union, Literal

import polars as pl
from polars.plugins import register_plugin_function

from polars_list_utils._internal import __version__ as __version__  # ty:ignore[unresolved-import]
from polars_list_utils._internal import fft_freqs, fft_freqs_linspace  # noqa: F401  # ty:ignore[unresolved-import]

root_path = Path(__file__).parent


def apply_fft(
    list_column: Union[pl.Expr, str, pl.Series],
    sample_rate: int,
    window: Optional[str] = None,
    bp_min: Optional[float] = None,
    bp_max: Optional[float] = None,
    bp_ord: Optional[int] = None,
    norm: Optional[str] = None,
    skip_fft: bool = False,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "sample_rate": sample_rate,
            "window": window,
            "bp_min": bp_min,
            "bp_max": bp_max,
            "bp_ord": bp_ord,
            "norm": norm,
            "skip_fft": skip_fft,
        },
        plugin_path=root_path,
        function_name="expr_fft",
        is_elementwise=True,
    )


def operate_scalar_on_list(
    list_column: Union[pl.Expr, str, pl.Series],
    scalar_column: Union[pl.Expr, str, pl.Series],
    operation: Literal["add", "sub", "mul", "div"],
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column, scalar_column],
        kwargs={
            "operation": operation,
        },
        plugin_path=root_path,
        function_name="expr_operate_scalar_on_list",
        is_elementwise=True,
    )


def interpolate_columns(
    x_data: Union[pl.Expr, str, pl.Series],
    y_data: Union[pl.Expr, str, pl.Series],
    x_interp: Union[pl.Expr, str, pl.Series],
) -> pl.Expr:
    return register_plugin_function(
        args=[x_data, y_data, x_interp],
        plugin_path=root_path,
        function_name="expr_interpolate_columns",
        is_elementwise=True,
    )


def aggregate_list_col_elementwise(
    list_column: Union[pl.Expr, str, pl.Series],
    aggregation: Literal["mean", "sum", "count", "product", "gmean"] = "mean",
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "aggregation": aggregation,
        },
        plugin_path=root_path,
        function_name="expr_aggregate_list_col_elementwise",
        is_elementwise=False,
        returns_scalar=True,
    )


def agg_of_range(
    list_column_y: Union[pl.Expr, str, pl.Series],
    list_column_x: Union[pl.Expr, str, pl.Series],
    aggregation: Literal["mean", "median", "sum", "count", "max", "min"],
    x_min: float,
    x_max: float,
    x_range_excluded: Optional[tuple[float, float]] = None,
    x_min_idx_offset: Optional[int] = None,
    x_max_idx_offset: Optional[int] = None,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column_y, list_column_x],
        kwargs={
            "aggregation": aggregation,
            "x_min": x_min,
            "x_max": x_max,
            "x_range_excluded": x_range_excluded,
            "x_min_idx_offset": x_min_idx_offset,
            "x_max_idx_offset": x_max_idx_offset,
        },
        plugin_path=root_path,
        function_name="expr_agg_of_range",
        is_elementwise=True,
    )


def mean_of_range(
    list_column_y: Union[pl.Expr, str, pl.Series],
    list_column_x: Union[pl.Expr, str, pl.Series],
    x_min: float,
    x_max: float,
    x_range_excluded: Optional[tuple[float, float]] = None,
    x_min_idx_offset: Optional[int] = None,
    x_max_idx_offset: Optional[int] = None,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column_y, list_column_x],
        kwargs={
            "aggregation": "mean",
            "x_min": x_min,
            "x_max": x_max,
            "x_range_excluded": x_range_excluded,
            "x_min_idx_offset": x_min_idx_offset,
            "x_max_idx_offset": x_max_idx_offset,
        },
        plugin_path=root_path,
        function_name="expr_agg_of_range",
        is_elementwise=True,
    )


def inner_join_lists(
    list_column_left: Union[pl.Expr, str, pl.Series],
    list_column_right: Union[pl.Expr, str, pl.Series],
    join_nulls: bool = False,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column_left, list_column_right],
        kwargs={
            "join_nulls": join_nulls,
        },
        plugin_path=root_path,
        function_name="expr_inner_join_lists",
        is_elementwise=True,
    )


def asof_join_lists(
    list_column_left: Union[pl.Expr, str, pl.Series],
    list_column_right: Union[pl.Expr, str, pl.Series],
    tolerance: Optional[float] = None,
    strategy: Literal["backward", "forward", "nearest"] = "backward",
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column_left, list_column_right],
        kwargs={
            "tolerance": tolerance,
            "strategy": strategy,
        },
        plugin_path=root_path,
        function_name="expr_asof_join_lists",
        is_elementwise=True,
    )


def list_zip(
    list_column_left: Union[pl.Expr, str, pl.Series],
    list_column_right: Union[pl.Expr, str, pl.Series],
    pad: bool = False,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column_left, list_column_right],
        kwargs={
            "pad": pad,
        },
        plugin_path=root_path,
        function_name="expr_list_zip",
        is_elementwise=True,
    )


def list_unzip(
    list_struct_column: Union[pl.Expr, str, pl.Series],
) -> pl.Expr:
    return register_plugin_function(
        args=[list_struct_column],
        plugin_path=root_path,
        function_name="expr_list_unzip",
        is_elementwise=True,
    )


def arg_sort_list(
    list_column: Union[pl.Expr, str, pl.Series],
    descending: bool = False,
    nulls_last: bool = False,
    maintain_order: bool = False,
    limit: Optional[int] = None,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "descending": descending,
            "nulls_last": nulls_last,
            "maintain_order": maintain_order,
            "limit": limit,
        },
        plugin_path=root_path,
        function_name="expr_arg_sort_list",
        is_elementwise=True,
    )
