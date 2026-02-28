import polars as pl
import polars_list_utils as polist


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
