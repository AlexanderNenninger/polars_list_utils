use crate::util::same_dtype;
use num_traits::{FromPrimitive, ToPrimitive};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct AggregateListKwargs {
    aggregation: String,
}

macro_rules! aggregate_list_native {
    ($fn_name:ident, $ca_method:ident, $val_ty:ty, $acc_ty:ty, $valid:expr) => {
        fn $fn_name(
            ca: &ListChunked,
            aggregation: &str,
            out_inner_dtype: &DataType,
        ) -> PolarsResult<Series> {
            let mut buckets: Vec<Option<$acc_ty>> = Vec::new();
            let mut counts: Vec<usize> = Vec::new();

            for row in ca.amortized_iter() {
                let Some(row) = row else {
                    continue;
                };

                let row = row.as_ref();
                let row = row.$ca_method()?;

                for (idx, val) in row.iter().enumerate() {
                    if idx >= buckets.len() {
                        buckets.resize(idx + 1, None);
                        counts.resize(idx + 1, 0);
                    }

                    if let Some(val) = val
                        && ($valid)(val)
                    {
                        let val: $acc_ty = val as $acc_ty;
                        match aggregation {
                            "mean" | "sum" => {
                                if let Some(bucket) = &mut buckets[idx] {
                                    *bucket += val;
                                } else {
                                    buckets[idx] = Some(val);
                                }
                                counts[idx] += 1;
                            }
                            "product" | "gmean" => {
                                if let Some(bucket) = &mut buckets[idx] {
                                    *bucket *= val;
                                } else {
                                    buckets[idx] = Some(val);
                                }
                                counts[idx] += 1;
                            }
                            "count" => {
                                counts[idx] += 1;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }

            let out = match aggregation {
                "mean" => buckets
                    .iter()
                    .zip(counts.iter())
                    .map(|(bucket, count)| {
                        if *count == 0 || bucket.is_none() {
                            None
                        } else {
                            Some(bucket.unwrap() / (*count as $acc_ty))
                        }
                    })
                    .collect::<Vec<Option<$acc_ty>>>(),
                "sum" => buckets,
                "count" => counts
                    .iter()
                    .map(|count| Some(*count as $acc_ty))
                    .collect::<Vec<Option<$acc_ty>>>(),
                "product" => buckets,
                "gmean" => buckets
                    .iter()
                    .zip(counts.iter())
                    .map(|(bucket, count)| {
                        if *count == 0 || bucket.is_none() {
                            None
                        } else {
                            let product = bucket.and_then(|b| b.to_f64());
                            match product {
                                Some(p) if p > 0.0 => {
                                    <$acc_ty as FromPrimitive>::from_f64(
                                        p.powf(1.0 / (*count as f64)),
                                    )
                                }
                                _ => None,
                            }
                        }
                    })
                    .collect::<Vec<Option<$acc_ty>>>(),
                _ => unreachable!(),
            };

            let out = Series::new(PlSmallStr::EMPTY, out)
                .implode()?
                .cast(&DataType::List(Box::new(out_inner_dtype.clone())))?;

            Ok(out.into())
        }
    };
}

aggregate_list_native!(agg_i8, i8, i8, i64, |_: i8| true);
aggregate_list_native!(agg_i16, i16, i16, i64, |_: i16| true);
aggregate_list_native!(agg_i32, i32, i32, i64, |_: i32| true);
aggregate_list_native!(agg_i64, i64, i64, i64, |_: i64| true);
aggregate_list_native!(agg_u8, u8, u8, u64, |_: u8| true);
aggregate_list_native!(agg_u16, u16, u16, u64, |_: u16| true);
aggregate_list_native!(agg_u32, u32, u32, u64, |_: u32| true);
aggregate_list_native!(agg_u64, u64, u64, u64, |_: u64| true);
aggregate_list_native!(agg_f32, f32, f32, f32, |x: f32| x.is_finite());
aggregate_list_native!(agg_f64, f64, f64, f64, |x: f64| x.is_finite());

/// Aggregate the elements, column-wise, of a `List` column.
///
/// The function raises an Error if:
/// * the aggregation method is not one of "mean", "sum", "count", "product", or "gmean"
///
/// The function dynamically expands its internal buffers when it encounters
/// longer sublists.
///
/// ## Parameters
/// - `aggregation`: The aggregation method to use. One of "mean", "sum", "count", "product", or "gmean".
///
/// ## Return value
/// New `List` column with the same inner dtype as the input.
#[polars_expr(output_type_func=same_dtype)]
fn expr_aggregate_list_col_elementwise(
    inputs: &[Series],
    kwargs: AggregateListKwargs,
) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;

    let valid_aggregations = ["mean", "sum", "count", "product", "gmean"];
    if !valid_aggregations.contains(&kwargs.aggregation.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "(aggregate_list_col_elementwise): Invalid aggregation method provided: {}. Must be one of [{}]",
                kwargs.aggregation,
                valid_aggregations.join(", "),
            )
            .into(),
        ));
    }

    let inner_dtype = ca.inner_dtype().clone();

    match inner_dtype {
        DataType::Int8 => agg_i8(ca, &kwargs.aggregation, &inner_dtype),
        DataType::Int16 => agg_i16(ca, &kwargs.aggregation, &inner_dtype),
        DataType::Int32 => agg_i32(ca, &kwargs.aggregation, &inner_dtype),
        DataType::Int64 => agg_i64(ca, &kwargs.aggregation, &inner_dtype),
        DataType::UInt8 => agg_u8(ca, &kwargs.aggregation, &inner_dtype),
        DataType::UInt16 => agg_u16(ca, &kwargs.aggregation, &inner_dtype),
        DataType::UInt32 => agg_u32(ca, &kwargs.aggregation, &inner_dtype),
        DataType::UInt64 => agg_u64(ca, &kwargs.aggregation, &inner_dtype),
        DataType::Float32 => agg_f32(ca, &kwargs.aggregation, &inner_dtype),
        DataType::Float64 => agg_f64(ca, &kwargs.aggregation, &inner_dtype),
        _ => Err(PolarsError::ComputeError(
            format!(
                "(aggregate_list_col_elementwise): unsupported list inner dtype {:?}",
                inner_dtype
            )
            .into(),
        )),
    }
}
