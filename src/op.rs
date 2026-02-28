use crate::util::{
    list_bool_dtype, list_f64_dtype, list_u32_dtype, struct_list_u32_dtype,
};
use interp::{InterpMode, interp_slice};
use polars::prelude::*;
use pyo3_polars::{
    derive::polars_expr, export::polars_core::utils::align_chunks_ternary,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct OperateScalarListKwargs {
    operation: String,
}

/// Apply a scalar operation elementwise to a `List` column.
///
/// The function raises an Error if:
/// * the operation is not one of "add", "sub", "mul" or "div"
///
/// ## Parameters
/// - `list_columns`: The `List` column to apply the operation to.
/// - `scalar_column`: The `Float64` scalar to apply the operation with.
/// - `operation`: The operation to apply. Must be one of "add", "sub", "mul" or "div".
///
/// ## Return value
/// New `List[f64]` column with the result of the operation.
#[polars_expr(output_type_func=list_f64_dtype)]
fn expr_operate_scalar_on_list(
    inputs: &[Series],
    kwargs: OperateScalarListKwargs,
) -> PolarsResult<Series> {
    let input_list = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let input_scalar = inputs[1].cast(&DataType::Float64)?;
    let list = input_list.list()?;
    let scalar = input_scalar.f64()?;

    let valid_operations = ["div", "mul", "add", "sub"];
    if !valid_operations.contains(&kwargs.operation.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "(operate_scalar_on_list): Invalid operation method provided: {}. Must be one of [{}]",
                kwargs.operation,
                valid_operations.join(", "),
            )
            .into(),
        ));
    }

    let out: ListChunked = list.zip_and_apply_amortized(scalar, |ca_list, ca_scalar| {
        if let (Some(ca_list), Some(ca_scalar)) = (ca_list, ca_scalar) {
            let list: Vec<Option<f64>> = ca_list
                .as_ref()
                .f64()
                .unwrap()
                .iter()
                .map(|val| {
                    if let Some(val) = val {
                        Some(match kwargs.operation.as_str() {
                            "div" => val / ca_scalar,
                            "mul" => val * ca_scalar,
                            "add" => val + ca_scalar,
                            "sub" => val - ca_scalar,
                            _ => unreachable!(),
                        })
                    } else {
                        None
                    }
                })
                .collect();

            Some(Series::new(PlSmallStr::EMPTY, list))
        } else {
            None
        }
    });

    Ok(out.into_series())
}

/// Interpolate columns to obtain `y_interp` from `x_data`, `x_interp` and `y_data`.
///
/// ## Parameters
/// - `x_data`: The `List` column containing the x-coords of the data.
/// - `y_data`: The `List` column containing the y-coords of the data.
/// - `x_interp`: The `List` column containing the x-coords of the interpolation.
///
/// ## Return value
/// New `List[f64]` column with the new y-coords of the interpolation, `y_interp`.
#[polars_expr(output_type_func=list_f64_dtype)]
fn expr_interpolate_columns(inputs: &[Series]) -> PolarsResult<Series> {
    let input_x_data = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let input_y_data = inputs[1].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let input_x_interp = inputs[2].cast(&DataType::List(Box::new(DataType::Float64)))?;

    let x_data = input_x_data.list()?;
    let y_data = input_y_data.list()?;
    let x_interp = input_x_interp.list()?;

    let (x_data, y_data, x_interp) = align_chunks_ternary(x_data, y_data, x_interp);

    let out: ListChunked = x_data
        .amortized_iter()
        .zip(y_data.amortized_iter())
        .zip(x_interp.amortized_iter())
        .map(|((x, y), x_interp)| {
            if let (Some(x), Some(y), Some(x_interp)) = (x, y, x_interp) {
                let x: Vec<f64> = x
                    .as_ref()
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|val| val.unwrap_or(f64::NAN))
                    .collect();

                let y: Vec<f64> = y
                    .as_ref()
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|val| val.unwrap_or(f64::NAN))
                    .collect();

                let x_interp: Vec<f64> = x_interp
                    .as_ref()
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|val| val.unwrap_or(f64::NAN))
                    .collect();

                let interpolated =
                    interp_slice(&x, &y, &x_interp, &InterpMode::FirstLast);

                Some(Series::new(PlSmallStr::EMPTY, interpolated))
            } else {
                None
            }
        })
        .collect();

    Ok(out.into_series())
}

#[derive(Deserialize)]
struct FillMissingListKwargs {
    method: String,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct InterpolateMissingListKwargs {
    mode: String,
}

#[derive(Deserialize)]
struct GapFlagsKwargs {
    #[serde(default = "default_min_gap")]
    min_gap: usize,
}

fn default_min_gap() -> usize {
    1
}

#[inline]
fn is_missing(v: Option<f64>) -> bool {
    match v {
        None => true,
        Some(x) => x.is_nan(),
    }
}

fn forward_fill_with_limit(
    values: &[Option<f64>],
    limit: Option<usize>,
) -> Vec<Option<f64>> {
    let mut out = Vec::with_capacity(values.len());
    let mut last_seen: Option<f64> = None;
    let mut gap_len = 0usize;

    for &v in values {
        if is_missing(v) {
            gap_len += 1;
            let fillable = limit.map(|l| gap_len <= l).unwrap_or(true);
            out.push(if fillable { last_seen } else { None });
        } else {
            last_seen = v;
            gap_len = 0;
            out.push(v);
        }
    }

    out
}

fn backward_fill_with_limit(
    values: &[Option<f64>],
    limit: Option<usize>,
) -> Vec<Option<f64>> {
    let mut out = vec![None; values.len()];
    let mut next_seen: Option<f64> = None;
    let mut gap_len = 0usize;

    for (idx, &v) in values.iter().enumerate().rev() {
        if is_missing(v) {
            gap_len += 1;
            let fillable = limit.map(|l| gap_len <= l).unwrap_or(true);
            out[idx] = if fillable { next_seen } else { None };
        } else {
            next_seen = v;
            gap_len = 0;
            out[idx] = v;
        }
    }

    out
}

fn interpolate_missing_values(
    values: &[Option<f64>],
    mode: &str,
) -> Vec<Option<f64>> {
    let n = values.len();
    let mut out = Vec::with_capacity(n);

    let known: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| v.filter(|x| !x.is_nan()).map(|x| (idx, x)))
        .collect();

    if known.is_empty() {
        return vec![None; n];
    }

    let known_idx: Vec<usize> = known.iter().map(|(idx, _)| *idx).collect();
    let known_vals: Vec<f64> = known.iter().map(|(_, v)| *v).collect();

    for (idx, &v) in values.iter().enumerate() {
        if !is_missing(v) {
            out.push(v);
            continue;
        }

        let pos = known_idx.partition_point(|&p| p < idx);
        let prev = (pos > 0).then_some(pos - 1);
        let next = (pos < known_idx.len()).then_some(pos);

        let interpolated = match mode {
            "linear" => {
                if let (Some(p), Some(nxt)) = (prev, next) {
                    let x0 = known_idx[p] as f64;
                    let y0 = known_vals[p];
                    let x1 = known_idx[nxt] as f64;
                    let y1 = known_vals[nxt];
                    if (x1 - x0).abs() < f64::EPSILON {
                        Some(y0)
                    } else {
                        let t = (idx as f64 - x0) / (x1 - x0);
                        Some(y0 + t * (y1 - y0))
                    }
                } else {
                    None
                }
            }
            "nearest" => match (prev, next) {
                (None, None) => None,
                (Some(p), None) => Some(known_vals[p]),
                (None, Some(nxt)) => Some(known_vals[nxt]),
                (Some(p), Some(nxt)) => {
                    let d_prev = idx - known_idx[p];
                    let d_next = known_idx[nxt] - idx;
                    if d_prev <= d_next {
                        Some(known_vals[p])
                    } else {
                        Some(known_vals[nxt])
                    }
                }
            },
            "first_last" => match (prev, next) {
                (None, None) => None,
                (Some(p), None) => Some(known_vals[p]),
                (None, Some(nxt)) => Some(known_vals[nxt]),
                (Some(p), Some(nxt)) => {
                    let x0 = known_idx[p] as f64;
                    let y0 = known_vals[p];
                    let x1 = known_idx[nxt] as f64;
                    let y1 = known_vals[nxt];
                    if (x1 - x0).abs() < f64::EPSILON {
                        Some(y0)
                    } else {
                        let t = (idx as f64 - x0) / (x1 - x0);
                        Some(y0 + t * (y1 - y0))
                    }
                }
            },
            _ => unreachable!(),
        };

        out.push(interpolated);
    }

    out
}

/// Fill missing values in each list row using forward or backward propagation.
/// Missing values include nulls and NaNs.
#[polars_expr(output_type_func=list_f64_dtype)]
fn expr_fill_missing_list(
    inputs: &[Series],
    kwargs: FillMissingListKwargs,
) -> PolarsResult<Series> {
    let valid_methods = ["forward", "backward"];
    if !valid_methods.contains(&kwargs.method.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "(fill_missing_list): invalid method '{}' ; expected one of [{}]",
                kwargs.method,
                valid_methods.join(", "),
            )
            .into(),
        ));
    }

    let input = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let list = input.list()?;

    let out: ListChunked = list
        .amortized_iter()
        .map(|row| {
            row.map(|row| {
                let vals: Vec<Option<f64>> = row.as_ref().f64().unwrap().iter().collect();
                let filled = match kwargs.method.as_str() {
                    "forward" => forward_fill_with_limit(&vals, kwargs.limit),
                    "backward" => backward_fill_with_limit(&vals, kwargs.limit),
                    _ => unreachable!(),
                };
                Series::new(PlSmallStr::EMPTY, filled)
            })
        })
        .collect();

    Ok(out.into_series())
}

/// Interpolate missing values in each list row.
/// Missing values include nulls and NaNs.
#[polars_expr(output_type_func=list_f64_dtype)]
fn expr_interpolate_missing_list(
    inputs: &[Series],
    kwargs: InterpolateMissingListKwargs,
) -> PolarsResult<Series> {
    let valid_modes = ["linear", "nearest", "first_last"];
    if !valid_modes.contains(&kwargs.mode.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "(interpolate_missing_list): invalid mode '{}' ; expected one of [{}]",
                kwargs.mode,
                valid_modes.join(", "),
            )
            .into(),
        ));
    }

    let input = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let list = input.list()?;

    let out: ListChunked = list
        .amortized_iter()
        .map(|row| {
            row.map(|row| {
                let vals: Vec<Option<f64>> = row.as_ref().f64().unwrap().iter().collect();
                let interpolated = interpolate_missing_values(&vals, &kwargs.mode);
                Series::new(PlSmallStr::EMPTY, interpolated)
            })
        })
        .collect();

    Ok(out.into_series())
}

/// Return per-element boolean flags indicating missing-value gaps.
/// A position is flagged `true` when it is part of a missing run with length
/// at least `min_gap`.
#[polars_expr(output_type_func=list_bool_dtype)]
fn expr_missing_gap_flags(
    inputs: &[Series],
    kwargs: GapFlagsKwargs,
) -> PolarsResult<Series> {
    if kwargs.min_gap == 0 {
        return Err(PolarsError::ComputeError(
            "(missing_gap_flags): min_gap must be >= 1".into(),
        ));
    }

    let input = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let list = input.list()?;

    let out: ListChunked = list
        .amortized_iter()
        .map(|row| {
            row.map(|row| {
                let vals: Vec<Option<f64>> = row.as_ref().f64().unwrap().iter().collect();
                let mut flags = vec![false; vals.len()];
                let mut i = 0usize;
                while i < vals.len() {
                    if is_missing(vals[i]) {
                        let start = i;
                        while i < vals.len() && is_missing(vals[i]) {
                            i += 1;
                        }
                        if i - start >= kwargs.min_gap {
                            for flag in &mut flags[start..i] {
                                *flag = true;
                            }
                        }
                    } else {
                        i += 1;
                    }
                }
                Series::new(PlSmallStr::EMPTY, flags)
            })
        })
        .collect();

    Ok(out.into_series())
}

#[derive(Deserialize, Clone, Copy)]
pub struct ArgsortListKwargs {
    descending: bool,
    nulls_last: bool,
    maintain_order: bool,
    limit: Option<u32>,
}

impl From<ArgsortListKwargs> for SortOptions {
    fn from(kwargs: ArgsortListKwargs) -> Self {
        SortOptions {
            descending: kwargs.descending,
            nulls_last: kwargs.nulls_last,
            maintain_order: kwargs.maintain_order,
            multithreaded: false, // Threading already happens at the chunk level.
            limit: kwargs.limit,  // Use the limit provided in the kwargs
        }
    }
}

/// Get the indices of the elements in list x, that would sort x.
/// The function returns a `List` column containing the indices of the elements in `x` that would sort `x`.
/// ## Parameters
/// - `x`: The `List` column containing the x-coords of the data.
/// - `descending`: Whether to sort in descending order.
///## Return value
/// New `List[u32]` column with the indices of the elements that would sort `x`.
#[polars_expr(output_type_func=list_u32_dtype)]
pub fn expr_arg_sort_list(
    inputs: &[Series],
    kwargs: ArgsortListKwargs,
) -> PolarsResult<Series> {
    let x = inputs[0].list()?;
    let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        x.name().clone(),
        x.len(),
        x.len(),
        DataType::UInt32,
    );
    for item in x.amortized_iter() {
        if let Some(item) = item {
            let item = item.as_ref();
            let argsorted = item.arg_sort(kwargs.into()).into_series();
            builder.append_series(&argsorted)?;
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish().into_series())
}
