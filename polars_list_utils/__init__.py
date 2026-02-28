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
    """Apply FFT-based processing to a list column.

    Args:
        list_column: Input list column expression.
        sample_rate: Sampling rate in Hz.
        window: Optional window function name.
        bp_min: Optional lower bandpass cutoff.
        bp_max: Optional upper bandpass cutoff.
        bp_ord: Optional bandpass filter order.
        norm: Optional normalization mode.
        skip_fft: If True, skips FFT and returns preprocessed signal.

    Returns:
        A Polars expression that evaluates to the processed list column.
    """
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
    """Apply a scalar operation to each element of a list column.

    Args:
        list_column: Input list column expression.
        scalar_column: Scalar column/expression used per row.
        operation: Arithmetic operation to apply.

    Returns:
        A Polars expression with the transformed list values.
    """
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
    """Interpolate y-values at new x-positions for each row.

    Args:
        x_data: Known x-coordinates as a list column.
        y_data: Known y-coordinates as a list column.
        x_interp: Target x-coordinates for interpolation.

    Returns:
        A Polars expression containing interpolated y-values.
    """
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
    """Aggregate list elements column-wise across rows.

    Args:
        list_column: Input list column expression.
        aggregation: Aggregation method to apply.

    Returns:
        A Polars expression with the aggregated list result.
    """
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
    """Aggregate y-values whose x-values fall within a range.

    Args:
        list_column_y: Y-value list column.
        list_column_x: X-value list column.
        aggregation: Aggregation method.
        x_min: Lower inclusive bound.
        x_max: Upper inclusive bound.
        x_range_excluded: Optional sub-range to exclude.
        x_min_idx_offset: Optional index offset from lower bound.
        x_max_idx_offset: Optional index offset from upper bound.

    Returns:
        A Polars expression with one aggregated value per row.
    """
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
    """Compute mean y-value inside an x-range for each row.

    Args:
        list_column_y: Y-value list column.
        list_column_x: X-value list column.
        x_min: Lower inclusive bound.
        x_max: Upper inclusive bound.
        x_range_excluded: Optional sub-range to exclude.
        x_min_idx_offset: Optional index offset from lower bound.
        x_max_idx_offset: Optional index offset from upper bound.

    Returns:
        A Polars expression with mean values per row.
    """
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
    """Compute row-wise inner-join index pairs between two list columns.

    Args:
        list_column_left: Left list column.
        list_column_right: Right list column.
        join_nulls: Whether null values should match null values.

    Returns:
        A Polars expression yielding a struct of left/right index lists.
    """
    return register_plugin_function(
        args=[list_column_left, list_column_right],
        kwargs={
            "join_nulls": join_nulls,
        },
        plugin_path=root_path,
        function_name="expr_inner_join_lists",
        is_elementwise=True,
    )


def left_join_lists(
    list_column_left: Union[pl.Expr, str, pl.Series],
    list_column_right: Union[pl.Expr, str, pl.Series],
    join_nulls: bool = False,
) -> pl.Expr:
    """Compute row-wise left-join index pairs between two list columns.

    Args:
        list_column_left: Left list column.
        list_column_right: Right list column.
        join_nulls: Whether null values should match null values.

    Returns:
        A Polars expression yielding a struct of left/right index lists.
    """
    return register_plugin_function(
        args=[list_column_left, list_column_right],
        kwargs={
            "join_nulls": join_nulls,
        },
        plugin_path=root_path,
        function_name="expr_left_join_lists",
        is_elementwise=True,
    )


def outer_join_lists(
    list_column_left: Union[pl.Expr, str, pl.Series],
    list_column_right: Union[pl.Expr, str, pl.Series],
    join_nulls: bool = False,
) -> pl.Expr:
    """Compute row-wise outer-join index pairs between two list columns.

    Args:
        list_column_left: Left list column.
        list_column_right: Right list column.
        join_nulls: Whether null values should match null values.

    Returns:
        A Polars expression yielding a struct of left/right index lists.
    """
    return register_plugin_function(
        args=[list_column_left, list_column_right],
        kwargs={
            "join_nulls": join_nulls,
        },
        plugin_path=root_path,
        function_name="expr_outer_join_lists",
        is_elementwise=True,
    )


def asof_join_lists(
    list_column_left: Union[pl.Expr, str, pl.Series],
    list_column_right: Union[pl.Expr, str, pl.Series],
    tolerance: Optional[float] = None,
    strategy: Literal["backward", "forward", "nearest"] = "backward",
) -> pl.Expr:
    """Compute row-wise asof-join index pairs between two sorted lists.

    Args:
        list_column_left: Left list column.
        list_column_right: Right list column.
        tolerance: Optional maximum allowed distance.
        strategy: Matching strategy: backward, forward, or nearest.

    Returns:
        A Polars expression yielding a struct of left/right index lists.
    """
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
    """Zip two list columns into a list of structs.

    Args:
        list_column_left: Left list column.
        list_column_right: Right list column.
        pad: If True, pad the shorter list with nulls.

    Returns:
        A Polars expression yielding a list of structs per row.
    """
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
    """Unzip a list-of-struct column into a struct-of-lists.

    Args:
        list_struct_column: Input list column with struct inner dtype.

    Returns:
        A Polars expression yielding a struct whose fields are list columns.
    """
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
    """Return row-wise argsort indices for list elements.

    Args:
        list_column: Input list column.
        descending: Sort descending if True.
        nulls_last: Place nulls at the end if True.
        maintain_order: Keep order among equal elements if True.
        limit: Optional number of top indices to return.

    Returns:
        A Polars expression yielding list indices for sorted order.
    """
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
