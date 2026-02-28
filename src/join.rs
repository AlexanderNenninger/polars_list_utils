use crate::util::{list_struct_dtype, struct_list_u32_dtype, struct_of_lists_dtype};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::cmp::Ordering;

#[derive(Deserialize, Default)]
struct ListZipKwargs {
    #[serde(default)]
    pad: bool,
}

fn pad_series_with_nulls(
    s: Series,
    target_len: usize,
) -> PolarsResult<Series> {
    if s.len() < target_len {
        s.extend_constant(AnyValue::Null, target_len - s.len())
    } else {
        Ok(s)
    }
}

/// Zip two list columns into a list of structs per row.
///
/// Each output element is a struct with fields named after the input columns.
/// If both inner lists are present, zipping is done up to the minimum of both
/// lengths by default. If `pad=true`, output length is the maximum of both
/// lengths and missing values are padded with nulls. If either row is null,
/// the output row is null.
#[polars_expr(output_type_func=list_struct_dtype)]
fn expr_list_zip(
    inputs: &[Series],
    kwargs: ListZipKwargs,
) -> PolarsResult<Series> {
    let left = inputs[0].list()?;
    let right = inputs[1].list()?;

    let left_name = left.name().clone();
    let right_name = right.name().clone();

    let mut rows: Vec<Option<Series>> = Vec::with_capacity(left.len());

    for (left_row, right_row) in left.amortized_iter().zip(right.amortized_iter()) {
        match (left_row, right_row) {
            (Some(left_row), Some(right_row)) => {
                let left_row = left_row.as_ref();
                let right_row = right_row.as_ref();
                let zipped_len = if kwargs.pad {
                    left_row.len().max(right_row.len())
                } else {
                    left_row.len().min(right_row.len())
                };

                let mut left_zipped = if kwargs.pad {
                    pad_series_with_nulls(left_row.clone(), zipped_len)?
                } else {
                    left_row.slice(0, zipped_len)
                };
                let mut right_zipped = if kwargs.pad {
                    pad_series_with_nulls(right_row.clone(), zipped_len)?
                } else {
                    right_row.slice(0, zipped_len)
                };
                left_zipped.rename(left_name.clone());
                right_zipped.rename(right_name.clone());

                let zipped_struct = StructChunked::from_series(
                    PlSmallStr::EMPTY,
                    zipped_len,
                    [left_zipped, right_zipped].iter(),
                )?
                .into_series();

                rows.push(Some(zipped_struct));
            }
            _ => rows.push(None),
        }
    }

    let out: ListChunked = rows.into_iter().collect();
    Ok(out.into_series())
}

/// Unzip a list-of-struct column into a struct-of-lists column.
///
/// For each row, this transforms `List[Struct{f1, f2, ...}]` into
/// `Struct{f1: List, f2: List, ...}`. If a row is null, each output field in
/// the struct is null for that row.
#[polars_expr(output_type_func=struct_of_lists_dtype)]
fn expr_list_unzip(inputs: &[Series]) -> PolarsResult<Series> {
    let zipped = inputs[0].list()?;

    let struct_fields = match zipped.inner_dtype() {
        DataType::Struct(fields) => fields.clone(),
        dt => {
            return Err(PolarsError::ComputeError(
                format!(
                    "(list_unzip): expected List[Struct] input dtype, got List[{dt:?}]"
                )
                .into(),
            ));
        }
    };

    let n_fields = struct_fields.len();
    let mut per_field_rows: Vec<Vec<Option<Series>>> = (0..n_fields)
        .map(|_| Vec::with_capacity(zipped.len()))
        .collect();

    for row in zipped.amortized_iter() {
        if let Some(row) = row {
            let row = row.as_ref();
            let struct_ca = row.struct_()?;
            let field_series = struct_ca.fields_as_series();
            for (rows, field_values) in per_field_rows.iter_mut().zip(field_series.iter())
            {
                rows.push(Some(field_values.clone()));
            }
        } else {
            for rows in &mut per_field_rows {
                rows.push(None);
            }
        }
    }

    let out_series: Vec<Series> = struct_fields
        .iter()
        .zip(per_field_rows.into_iter())
        .map(|(field, rows)| {
            let mut list_series = ListChunked::from_iter(rows).into_series();
            list_series.rename(field.name.clone());
            list_series
        })
        .collect();

    let out = StructChunked::from_series(
        zipped.name().clone(),
        zipped.len(),
        out_series.iter(),
    )?;

    Ok(out.into_series())
}

#[derive(Deserialize)]
struct ListInnerJoinKwargs {
    join_nulls: bool,
}

/// Get the indices of the elements in list x, that are equal to the corresponding elements in list y.
/// The function returns a `List` column containing the indices of the elements in `x` that are equal to the corresponding elements in `y`.
/// This can be used to join two `List` columns on the values of the inner lists, by first getting the indices of the matching values and then
/// using those indices to gather the corresponding values from another column.
/// ## Parameters
/// - `x`: The `List` column containing the x-coords of the data.
/// - `y`: The `List` column containing the y-coords of the data.
/// - `join_nulls`: Whether to consider null values as equal when performing the join. If true, null values will be treated as equal and their indices will be included in the output. If false, null values will not be treated as equal and their indices will not be included in the output.
///## Return value
/// New `List[u32]` column with the indices of the matching elements.
#[polars_expr(output_type_func=struct_list_u32_dtype)]
fn expr_inner_join_lists(
    inputs: &[Series],
    kwargs: ListInnerJoinKwargs,
) -> PolarsResult<Series> {
    let x = inputs[0].list()?;
    let y = inputs[1].list()?;

    let mut left_indices_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        x.name().clone(),
        x.len(),
        x.len(),
        DataType::UInt32,
    );
    let mut right_indices_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        y.name().clone(),
        y.len(),
        y.len(),
        DataType::UInt32,
    );

    for (x, y) in x.amortized_iter().zip(y.amortized_iter()) {
        if let (Some(x), Some(y)) = (x, y) {
            let x = x.as_ref();
            let y = y.as_ref();
            if let Some(((left_keys, right_keys), _sorted)) = x
                .hash_join_inner(y, JoinValidation::default(), kwargs.join_nulls)
                .ok()
            {
                left_indices_builder.append_slice(&left_keys);
                right_indices_builder.append_slice(&right_keys);
            }
        } else {
            left_indices_builder.append_null();
            right_indices_builder.append_null();
        }
    }

    let out = StructChunked::from_series(
        x.name().clone(),
        x.len(),
        [
            left_indices_builder.finish().into_series(),
            right_indices_builder.finish().into_series(),
        ]
        .iter(),
    )?;

    Ok(out.into_series())
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum AsofJoinStrategy {
    Backward,
    Forward,
    Nearest,
}

impl Default for AsofJoinStrategy {
    fn default() -> Self {
        Self::Backward
    }
}

#[derive(Deserialize, Clone, Copy)]
struct AsofJoinListsKwargs {
    tolerance: Option<f64>,
    #[serde(default)]
    strategy: AsofJoinStrategy,
}

fn any_value_cmp(
    left: &AnyValue<'_>,
    right: &AnyValue<'_>,
) -> PolarsResult<Ordering> {
    match left.partial_cmp(right) {
        Some(ord) => Ok(ord),
        None => Err(PolarsError::ComputeError(
            format!(
                "(asof_join_lists): values are not comparable for dtype pair ({:?}, {:?})",
                left.dtype(),
                right.dtype(),
            )
            .into(),
        )),
    }
}

fn any_value_to_f64(v: &AnyValue<'_>) -> Option<f64> {
    match v {
        AnyValue::Float32(x) => Some(*x as f64),
        AnyValue::Float64(x) => Some(*x),
        AnyValue::Int8(x) => Some(*x as f64),
        AnyValue::Int16(x) => Some(*x as f64),
        AnyValue::Int32(x) => Some(*x as f64),
        AnyValue::Int64(x) => Some(*x as f64),
        AnyValue::UInt8(x) => Some(*x as f64),
        AnyValue::UInt16(x) => Some(*x as f64),
        AnyValue::UInt32(x) => Some(*x as f64),
        AnyValue::UInt64(x) => Some(*x as f64),
        AnyValue::Date(x) => Some(*x as f64),
        AnyValue::Datetime(x, _, _) => Some(*x as f64),
        AnyValue::Duration(x, _) => Some(*x as f64),
        AnyValue::Time(x) => Some(*x as f64),
        _ => None,
    }
}

fn any_value_abs_diff_f64(
    left: &AnyValue<'_>,
    right: &AnyValue<'_>,
    dtype: &DataType,
) -> PolarsResult<f64> {
    match (any_value_to_f64(left), any_value_to_f64(right)) {
        (Some(l), Some(r)) => Ok((l - r).abs()),
        _ => Err(PolarsError::ComputeError(
            format!(
                "(asof_join_lists): distance-based matching is only supported for numeric or temporal list dtypes, got {:?}",
                dtype
            )
            .into(),
        )),
    }
}

fn choose_asof_match_index(
    left_val: &AnyValue<'_>,
    y_valid: &[(u32, AnyValue<'static>)],
    y_pos: &mut usize,
    strategy: AsofJoinStrategy,
    dtype: &DataType,
) -> PolarsResult<Option<usize>> {
    match strategy {
        AsofJoinStrategy::Backward => {
            while *y_pos < y_valid.len()
                && matches!(
                    any_value_cmp(&y_valid[*y_pos].1, left_val)?,
                    Ordering::Less | Ordering::Equal
                )
            {
                *y_pos += 1;
            }

            Ok(if *y_pos > 0 { Some(*y_pos - 1) } else { None })
        }
        AsofJoinStrategy::Forward => {
            while *y_pos < y_valid.len()
                && any_value_cmp(&y_valid[*y_pos].1, left_val)? == Ordering::Less
            {
                *y_pos += 1;
            }

            Ok((*y_pos < y_valid.len()).then_some(*y_pos))
        }
        AsofJoinStrategy::Nearest => {
            while *y_pos < y_valid.len()
                && any_value_cmp(&y_valid[*y_pos].1, left_val)? == Ordering::Less
            {
                *y_pos += 1;
            }

            let prev = if *y_pos > 0 { Some(*y_pos - 1) } else { None };
            let next = (*y_pos < y_valid.len()).then_some(*y_pos);

            match (prev, next) {
                (None, None) => Ok(None),
                (Some(p), None) => Ok(Some(p)),
                (None, Some(n)) => Ok(Some(n)),
                (Some(p), Some(n)) => {
                    let prev_dist =
                        any_value_abs_diff_f64(left_val, &y_valid[p].1, dtype)?;
                    let next_dist =
                        any_value_abs_diff_f64(left_val, &y_valid[n].1, dtype)?;
                    if prev_dist <= next_dist {
                        Ok(Some(p))
                    } else {
                        Ok(Some(n))
                    }
                }
            }
        }
    }
}

fn compute_asof_join_indices(
    x: &Series,
    y: &Series,
    tolerance: Option<f64>,
    strategy: AsofJoinStrategy,
) -> PolarsResult<(Vec<u32>, Vec<u32>)> {
    if x.dtype() != y.dtype() {
        return Err(PolarsError::ComputeError(
            format!(
                "(asof_join_lists): list element dtypes must match, got left={:?}, right={:?}",
                x.dtype(),
                y.dtype()
            )
            .into(),
        ));
    }

    // Keep original right-side indices, skipping nulls.
    let y_valid: Vec<(u32, AnyValue<'static>)> = y
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| (!v.is_null()).then_some((idx as u32, v.into_static())))
        .collect();

    let mut left_keys = Vec::<u32>::new();
    let mut right_keys = Vec::<u32>::new();
    let mut y_pos = 0usize;

    for (left_idx, left_val) in x.iter().enumerate() {
        if left_val.is_null() {
            continue;
        }

        let Some(match_pos) = choose_asof_match_index(
            &left_val,
            &y_valid,
            &mut y_pos,
            strategy,
            x.dtype(),
        )?
        else {
            continue;
        };

        let (right_idx, right_val) = &y_valid[match_pos];
        let within_tolerance = tolerance
            .map(|tol| {
                any_value_abs_diff_f64(&left_val, right_val, x.dtype()).map(|d| d <= tol)
            })
            .transpose()?
            .unwrap_or(true);

        if within_tolerance {
            left_keys.push(left_idx as u32);
            right_keys.push(*right_idx);
        }
    }

    Ok((left_keys, right_keys))
}

/// Get the indices that would perform an asof join between two list columns.
/// The function returns a `Struct` column containing two `List[u32]` columns,
/// one with the indices of the left list and one with the indices of the right list that would match in an asof join.
/// ## Parameters
/// - `x`: The `List` column containing the x-coords of the data.
/// - `y`: The `List` column containing the y-coords of the data.
/// - `tolerance`: The maximum difference between the values in `x` and `y` for them to be considered a match in the asof join.
/// - `strategy`: Matching direction. One of `"backward"`, `"forward"`, `"nearest"`.
///## Return value
/// New `Struct` column with two `List[u32]` columns containing the indices of the matching elements in `x` and `y`.
#[polars_expr(output_type_func=struct_list_u32_dtype)]
pub fn expr_asof_join_lists(
    inputs: &[Series],
    kwargs: AsofJoinListsKwargs,
) -> PolarsResult<Series> {
    let x = inputs[0].list()?;
    let y = inputs[1].list()?;
    let tolerance = if let Some(tol) = kwargs.tolerance {
        if tol < 0.0 {
            return Err(PolarsError::ComputeError(
                "(asof_join_lists): tolerance must be >= 0".into(),
            ));
        }
        Some(tol)
    } else {
        None
    };
    let strategy = kwargs.strategy;

    let mut left_indices_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        x.name().clone(),
        x.len(),
        x.len(),
        DataType::UInt32,
    );
    let mut right_indices_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        y.name().clone(),
        y.len(),
        y.len(),
        DataType::UInt32,
    );

    for (x, y) in x.amortized_iter().zip(y.amortized_iter()) {
        if let (Some(x), Some(y)) = (x, y) {
            let x = x.as_ref();
            let y = y.as_ref();
            let (left_keys, right_keys) =
                compute_asof_join_indices(x, y, tolerance, strategy)?;

            left_indices_builder.append_slice(&left_keys);
            right_indices_builder.append_slice(&right_keys);
        } else {
            left_indices_builder.append_null();
            right_indices_builder.append_null();
        }
    }

    let out = StructChunked::from_series(
        x.name().clone(),
        x.len(),
        [
            left_indices_builder.finish().into_series(),
            right_indices_builder.finish().into_series(),
        ]
        .iter(),
    )?;

    Ok(out.into_series())
}
