use chrono::{DateTime, NaiveDateTime, Utc};

// Import CategoricalOrdering and SortOptions
use polars::datatypes::{CategoricalOrdering, DataType, TimeUnit};
use polars::prelude::*; // Includes StringChunkedBuilder, SortOptions etc.
mod query_stats;
use query_stats::QueryStats;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::time::{Duration as StdDuration, Instant}; // Renamed to avoid conflict
use uuid::Uuid;

// --- Struct Definitions (Unchanged) ---
#[derive(Serialize, Deserialize, Debug, Clone)]
struct LogSource {
    ip: String,
    host: String,
    region: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct UserMetrics {
    login_time_ms: i64,
    clicks: i64,
    active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct User {
    id: String,
    session_id: String,
    metrics: UserMetrics,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Answer {
    #[serde(rename = "nxDomain")]
    nx_domain: bool,
    response_time_ms: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LogRecord {
    doc_id: i64,
    timestamp: String,
    level: String,
    message: String,
    source: LogSource,
    user: User,
    payload_size: i64,
    tags: Vec<String>,
    answers: Vec<Answer>,
    processed: bool,
}

// --- Data Generation (Unchanged) ---
fn generate_random_log_record(i: usize, base_time: DateTime<Utc>) -> LogRecord {
    let mut rng = rand::thread_rng();
    let levels = ["info", "warn", "error", "debug", "trace"];
    let regions = [
        "us-east-1",
        "eu-west-1",
        "eu-west-2",
        "ap-south-1",
        "us-west-2",
    ];
    let hosts = (1..=20)
        .map(|n| format!("server-{}.region.local", n))
        .collect::<Vec<_>>();
    let offset_ms = rng.gen_range(-30000..30000);
    let timestamp = base_time + chrono::Duration::milliseconds(offset_ms);
    let answers_len = rng.gen_range(0..=3);
    let answers = (0..answers_len)
        .map(|_| Answer {
            nx_domain: rng.gen_bool(0.3),
            response_time_ms: rng.gen_range(5..150),
        })
        .collect::<Vec<_>>();
    LogRecord {
        doc_id: i as i64,
        timestamp: timestamp.to_rfc3339(),
        level: levels[rng.gen_range(0..levels.len())].to_string(),
        message: format!("Log message {} for record {}", Uuid::new_v4(), i),
        source: LogSource {
            ip: format!("10.0.{}.{}", rng.gen_range(1..255), rng.gen_range(1..255)),
            host: hosts[rng.gen_range(0..hosts.len())].clone(),
            region: regions[rng.gen_range(0..regions.len())].to_string(),
        },
        user: User {
            id: format!("user_{}", rng.gen_range(1000..50000)),
            session_id: Uuid::new_v4().to_string(),
            metrics: UserMetrics {
                login_time_ms: rng.gen_range(10..1500),
                clicks: rng.gen_range(0..100),
                active: rng.gen_bool(0.75),
            },
        },
        payload_size: rng.gen_range(50..20_480),
        tags: (0..rng.gen_range(1..8))
            .map(|_| format!("tag_{}", rng.gen_range(1..50)))
            .collect::<Vec<_>>(),
        answers,
        processed: rng.gen_bool(0.9),
    }
}

// --- OPTIMIZED DataFrame Creation (Fixed) ---
fn create_dataframe_from_records_optimized(records: &[LogRecord]) -> PolarsResult<DataFrame> {
    // --- Primitive and Simple String Columns ---
    let doc_ids: Vec<i64> = records.iter().map(|r| r.doc_id).collect();
    let levels: Vec<&str> = records.iter().map(|r| r.level.as_str()).collect();
    let messages: Vec<&str> = records.iter().map(|r| r.message.as_str()).collect();
    let source_ips: Vec<&str> = records.iter().map(|r| r.source.ip.as_str()).collect();
    let source_hosts: Vec<&str> = records.iter().map(|r| r.source.host.as_str()).collect();
    let source_regions: Vec<&str> = records.iter().map(|r| r.source.region.as_str()).collect();
    let user_ids: Vec<&str> = records.iter().map(|r| r.user.id.as_str()).collect();
    let user_session_ids: Vec<&str> = records.iter().map(|r| r.user.session_id.as_str()).collect();
    let user_metrics_login_time: Vec<i64> = records
        .iter()
        .map(|r| r.user.metrics.login_time_ms)
        .collect();
    let user_metrics_clicks: Vec<i64> = records.iter().map(|r| r.user.metrics.clicks).collect();
    let user_metrics_active: Vec<bool> = records.iter().map(|r| r.user.metrics.active).collect();
    let payload_sizes: Vec<i64> = records.iter().map(|r| r.payload_size).collect();
    let processed: Vec<bool> = records.iter().map(|r| r.processed).collect();

    // --- Timestamp Column (Parse to Native Datetime) ---
    let timestamps_naive: Vec<Option<NaiveDateTime>> = records
        .iter()
        .map(|r| {
            DateTime::parse_from_rfc3339(&r.timestamp)
                .ok()
                .map(|dt| dt.naive_utc())
        })
        .collect();
    let timestamps_series = Series::new("timestamp".into(), timestamps_naive)
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;

    // --- Tags Column (List<String>) ---
    let tags_series_inner: Vec<Series> = records
        .iter()
        .map(|r| Series::new("".into(), &r.tags)) // Inner Series for each record's tags
        .collect();
    let tags_series = Series::new("tags".into(), &tags_series_inner); // Create List<String> Series

    // --- Answers Column (List<Struct>) ---
    // Revised approach: Create a Vec<Series>, where each Series is a StructChunked for one record
    let answers_series_inner: Vec<Series> = records
        .iter()
        .map(|r| {
            // Create inner columns for *this* record's answers
            let nx_domains: Vec<bool> = r.answers.iter().map(|a| a.nx_domain).collect();
            let response_times: Vec<i64> = r.answers.iter().map(|a| a.response_time_ms).collect();

            let nx_domain_ca = BooleanChunked::from_slice("nx_domain".into(), &nx_domains);
            let response_time_ca =
                Int64Chunked::from_slice("response_time_ms".into(), &response_times);

            // Create a Struct Series for this record
            StructChunked::from_series(
                "".into(), // Name for inner struct series doesn't matter here
                &[nx_domain_ca.into_series(), response_time_ca.into_series()],
            )
            .map(|sc| sc.into_series()) // Convert Result<StructChunked> to Result<Series>
        })
        .collect::<PolarsResult<Vec<Series>>>()?; // Collect results, handling potential errors

    // Create the final List<Struct> series from the Vec<Series>
    let answers_series = Series::new("answers".into(), &answers_series_inner);

    // --- Assemble DataFrame ---
    DataFrame::new(vec![
        Series::new("doc_id".into(), doc_ids),
        timestamps_series,
        Series::new("level".into(), levels),
        Series::new("message".into(), messages),
        Series::new("source_ip".into(), source_ips),
        Series::new("source_host".into(), source_hosts),
        Series::new("source_region".into(), source_regions),
        Series::new("user_id".into(), user_ids),
        Series::new("user_session_id".into(), user_session_ids),
        Series::new("user_metrics_login_time_ms".into(), user_metrics_login_time),
        Series::new("user_metrics_clicks".into(), user_metrics_clicks),
        Series::new("user_metrics_active".into(), user_metrics_active),
        Series::new("payload_size".into(), payload_sizes),
        tags_series,
        answers_series, // Use the List<Struct> series created above
        Series::new("processed".into(), processed),
    ])
}

// --- OPTIMIZED Parquet Writing (Fixed) ---
fn write_records_to_single_parquet_optimized(
    records: Vec<LogRecord>,
    file_path: &str,
    compression: ParquetCompression,
    row_group_size: Option<usize>,
) -> PolarsResult<()> {
    if records.is_empty() {
        return Ok(());
    }
    let start_time = Instant::now();
    log::info!(
        "Writing {} records to single Parquet file {} (Optimized)...",
        records.len(),
        file_path
    );

    // 1. Create DataFrame using the optimized function
    let mut df = create_dataframe_from_records_optimized(&records)?;
    log::info!(
        "Created DataFrame with optimized types in {:?}",
        start_time.elapsed()
    );
    let create_df_time = Instant::now();

    // 2. Apply Categorical Casting (Fixed: Added CategoricalOrdering)
    let categorical_dtype = DataType::Categorical(None, CategoricalOrdering::Physical);
    df.try_apply("level", |s| s.cast(&categorical_dtype))?;
    df.try_apply("source_region", |s| s.cast(&categorical_dtype))?;
    df.try_apply("source_host", |s| s.cast(&categorical_dtype))?;
    log::info!(
        "Applied Categorical casts in {:?}",
        create_df_time.elapsed()
    );
    let cast_time = Instant::now();

    // 3. Sort DataFrame (Fixed: Use SortOptions::default() for single column)
    df = df.sort(["doc_id"], SortMultipleOptions::default())?;
    log::info!("Sorted DataFrame by doc_id in {:?}", cast_time.elapsed());
    let sort_time = Instant::now();

    // 4. Write to Parquet
    let file = File::create(file_path)?;
    let buf_writer = BufWriter::new(file);

    ParquetWriter::new(buf_writer)
        .with_compression(compression)
        .with_statistics(StatisticsOptions::full())
        .with_row_group_size(row_group_size)
        .finish(&mut df)?;

    let duration = start_time.elapsed();
    // Note: df height might be slightly different if creation failed partially, but generally ok
    log::info!(
        "Successfully wrote {} records to {} in {:?} (Write took {:?})",
        records.len(), // Use original record count for logging consistency
        file_path,
        duration,
        sort_time.elapsed()
    );
    Ok(())
}

// --- Helper for Field Names (Unchanged) ---
fn field_name_to_column(field_name: &str) -> String {
    field_name.replace('.', "_")
}

// --- Result Struct (Unchanged) ---
#[derive(Debug)]
struct FieldValueResult {
    value_map: HashMap<String, Vec<i64>>,
}

// --- REFACTORED get_field_values_by_doc_ids (Fixed) ---
fn get_field_values_by_doc_ids_refactored(
    field_name: &str,
    doc_ids: &[i64],
    file_path: &str,
    low_memory: bool,
) -> PolarsResult<(FieldValueResult, QueryStats)> {
    let mut stats = QueryStats::new(
        "get_field_values_by_doc_ids",
        field_name,
        Some(doc_ids.len()),
    );
    let overall_start = Instant::now();

    // Setup phase
    let args = time_section!(stats, setup, {
        println!(
            "Querying get_field_values_by_doc_ids for field '{}' with {} doc_ids",
            field_name,
            doc_ids.len()
        );
        ScanArgsParquet {
            low_memory,
            ..Default::default()
        }
    });

    let lf = LazyFrame::scan_parquet(file_path, args)?;
    let column_name = field_name_to_column(field_name);

    // Filter creation phase
    let filter_df = time_section!(stats, filter_creation, {
        df! (
            "doc_id" => doc_ids,
        )?
        .lazy()
        .sort(["doc_id"], SortMultipleOptions::default())
    });

    // Join operation phase
    let filtered_lf = time_section!(stats, join_operation, {
        lf.join(
            filter_df,
            [col("doc_id")],
            [col("doc_id")],
            JoinArgs::new(JoinType::Inner),
        )
        .select([col(&column_name), col("doc_id")])
    });

    // Collect phase
    let df = time_section!(stats, collect, { filtered_lf.collect()? });
    stats.set_result_rows(df.height());

    // Processing phase
    let value_map = time_section!(stats, processing, {
        let mut map: HashMap<String, Vec<i64>> = HashMap::new();
        if df.height() > 0 {
            let field_col = df.column(&column_name)?;
            let ids_col = df.column("doc_id")?;
            let ids_ca = ids_col.i64()?;

            match field_col.dtype() {
                DataType::Categorical(_, _) => {
                    let field_ca = field_col.categorical()?;
                    field_ca
                        .iter_str()
                        .zip(ids_ca.into_iter())
                        .for_each(|(key_opt, id_opt)| {
                            if let (Some(key), Some(id)) = (key_opt, id_opt) {
                                map.entry(key.to_string()).or_default().push(id);
                            }
                        });
                }
                DataType::String => {
                    let field_ca = field_col.str()?;
                    field_ca
                        .into_iter()
                        .zip(ids_ca.into_iter())
                        .for_each(|(key_opt, id_opt)| {
                            if let (Some(key), Some(id)) = (key_opt, id_opt) {
                                map.entry(key.to_string()).or_default().push(id);
                            }
                        });
                }
                DataType::Boolean => {
                    let field_ca = field_col.bool()?;
                    field_ca
                        .into_iter()
                        .zip(ids_ca.into_iter())
                        .for_each(|(key_opt, id_opt)| {
                            if let (Some(key_bool), Some(id)) = (key_opt, id_opt) {
                                let key_str = key_bool.to_string();
                                map.entry(key_str).or_default().push(id);
                            }
                        });
                }
                other_type => {
                    log::warn!(
                        "Grouping column '{}' has unexpected type: {:?}. Trying AnyValue conversion.",
                        column_name,
                        other_type
                    );
                    for i in 0..df.height() {
                        if let (Ok(key_any), Ok(id_any)) = (field_col.get(i), ids_col.get(i)) {
                            if let AnyValue::Int64(id) = id_any {
                                let key_str = key_any.to_string().trim_matches('"').to_string();
                                map.entry(key_str).or_default().push(id);
                            }
                        }
                    }
                }
            }
        }
        map
    });

    // Update total time
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();

    // Print summary
    stats.print_summary();

    Ok((FieldValueResult { value_map }, stats))
}

// --- REFACTORED get_field_values (Fixed) ---
fn get_field_values_refactored(
    field_name: &str,
    file_path: &str,
    low_memory: bool,
) -> PolarsResult<(FieldValueResult, QueryStats)> {
    println!(
        "Querying get_field_values_refactored for field '{}'",
        field_name
    );
    let mut stats = QueryStats::new("get_field_values_refactored", field_name, None);
    let overall_start = Instant::now();

    // Setup phase: create scan arguments with timing
    let args = time_section!(stats, setup, {
        ScanArgsParquet {
            low_memory,
            ..Default::default()
        }
    });
    let lf = LazyFrame::scan_parquet(file_path, args)?;
    let column_name = field_name_to_column(field_name);

    // Directly perform the select operation (no separate "select" timing available)
    let result_lf = lf.select([col(&column_name), col("doc_id")]);

    // Collect phase: materialize the DataFrame with timing
    let df = time_section!(stats, collect, { result_lf.collect()? });
    stats.set_result_rows(df.height());
    println!(
        "Collected DataFrame for get_field_values_refactored, height: {}",
        df.height()
    );

    // Processing phase (grouping): group 'field_name' values with timing
    let value_map = time_section!(stats, processing, {
        let mut map: HashMap<String, Vec<i64>> = HashMap::new();
        if df.height() > 0 {
            let field_col = df.column(&column_name)?;
            let ids_col = df.column("doc_id")?;
            let ids_ca = ids_col.i64()?;
            match field_col.dtype() {
                DataType::Categorical(_, _) => {
                    let field_ca = field_col.categorical()?;
                    field_ca
                        .iter_str()
                        .zip(ids_ca.into_iter())
                        .for_each(|(key_opt, id_opt)| {
                            if let (Some(key), Some(id)) = (key_opt, id_opt) {
                                map.entry(key.to_string()).or_default().push(id);
                            }
                        });
                }
                DataType::String => {
                    let field_ca = field_col.str()?;
                    field_ca
                        .into_iter()
                        .zip(ids_ca.into_iter())
                        .for_each(|(key_opt, id_opt)| {
                            if let (Some(key), Some(id)) = (key_opt, id_opt) {
                                map.entry(key.to_string()).or_default().push(id);
                            }
                        });
                }
                DataType::Boolean => {
                    let field_ca = field_col.bool()?;
                    field_ca
                        .into_iter()
                        .zip(ids_ca.into_iter())
                        .for_each(|(key_opt, id_opt)| {
                            if let (Some(key_bool), Some(id)) = (key_opt, id_opt) {
                                let key_str = key_bool.to_string();
                                map.entry(key_str).or_default().push(id);
                            }
                        });
                }
                other_type => {
                    log::warn!(
                        "Grouping column '{}' has unexpected type: {:?}. Trying AnyValue conversion.",
                        column_name,
                        other_type
                    );
                    for i in 0..df.height() {
                        if let (Ok(key_any), Ok(id_any)) = (field_col.get(i), ids_col.get(i)) {
                            if let AnyValue::Int64(id) = id_any {
                                let key_str = key_any.to_string().trim_matches('"').to_string();
                                map.entry(key_str).or_default().push(id);
                            }
                        }
                    }
                }
            }
        }
        map
    });

    // Update overall timing and memory usage
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();
    stats.print_summary();

    Ok((FieldValueResult { value_map }, stats))
}

// --- Numeric Stats Struct (Unchanged) ---
#[derive(Debug, Clone, PartialEq)]
pub struct NumericStats {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub avg: Option<f64>,
}

// --- REFACTORED get_numeric_stats_by_doc_ids (Fixed) ---
fn get_numeric_stats_by_doc_ids_refactored(
    field_name: &str,
    doc_ids: &[i64],
    file_path: &str,
    low_memory: bool,
) -> PolarsResult<(NumericStats, QueryStats)> {
    let mut stats = QueryStats::new(
        "get_numeric_stats_by_doc_ids",
        field_name,
        Some(doc_ids.len()),
    );
    let overall_start = Instant::now();

    // Setup phase
    let args = time_section!(stats, setup, {
        println!(
            "Querying get_numeric_stats_by_doc_ids for field '{}' with {} doc_ids",
            field_name,
            doc_ids.len()
        );
        ScanArgsParquet {
            low_memory,
            ..Default::default()
        }
    });

    let lf = LazyFrame::scan_parquet(file_path, args)?;
    let column_name = field_name_to_column(field_name);

    // Filter creation phase
    let filter_df = time_section!(stats, filter_creation, {
        df! (
            "doc_id" => doc_ids,
        )?
        .lazy()
        .sort(["doc_id"], SortMultipleOptions::default())
    });

    // Join operation phase
    let filtered_lf = time_section!(stats, join_operation, {
        lf.join(
            filter_df,
            [col("doc_id")],
            [col("doc_id")],
            JoinArgs::new(JoinType::Inner),
        )
    });

    // Define aggregation expressions and collect
    let numeric_expr_int = col(&column_name);
    let numeric_expr_float = numeric_expr_int.clone().cast(DataType::Float64);

    let result_lf = filtered_lf.select([
        numeric_expr_int
            .clone()
            .min()
            .cast(DataType::Float64)
            .alias("min"),
        numeric_expr_int
            .clone()
            .max()
            .cast(DataType::Float64)
            .alias("max"),
        numeric_expr_float.mean().alias("avg"),
    ]);

    // Collect phase
    let df = time_section!(stats, collect, { result_lf.collect()? });
    stats.set_result_rows(df.height());

    // Processing phase
    let numeric_stats = time_section!(stats, processing, {
        if df.height() == 0 {
            NumericStats {
                min: None,
                max: None,
                avg: None,
            }
        } else {
            let min_val = df.column("min")?.f64()?.get(0);
            let max_val = df.column("max")?.f64()?.get(0);
            let avg_val = df.column("avg")?.f64()?.get(0);
            NumericStats {
                min: min_val,
                max: max_val,
                avg: avg_val,
            }
        }
    });

    // Update total time
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();

    // Print summary
    stats.print_summary();

    Ok((numeric_stats, stats))
}

// --- REFACTORED get_numeric_stats (Unchanged from previous fix) ---
fn get_numeric_stats_refactored(
    field_name: &str,
    file_path: &str,
    low_memory: bool,
) -> PolarsResult<(NumericStats, QueryStats)> {
    println!(
        "Querying get_numeric_stats_refactored for field '{}'",
        field_name
    );
    // Initialize QueryStats instrumentation
    let mut stats = QueryStats::new("get_numeric_stats_refactored", field_name, None);
    let overall_start = Instant::now();

    // Setup phase: create scan arguments with timing
    let args = time_section!(stats, setup, {
        ScanArgsParquet {
            low_memory,
            ..Default::default()
        }
    });
    let lf = LazyFrame::scan_parquet(file_path, args)?;
    let column_name = field_name_to_column(field_name);

    // Construct the aggregation expressions for numeric stats
    let numeric_expr_int = col(&column_name);
    let numeric_expr_float = numeric_expr_int.clone().cast(DataType::Float64);
    let result_lf = lf.select([
        numeric_expr_int
            .clone()
            .min()
            .cast(DataType::Float64)
            .alias("min"),
        numeric_expr_int
            .clone()
            .max()
            .cast(DataType::Float64)
            .alias("max"),
        numeric_expr_float.mean().alias("avg"),
    ]);

    // Collect phase: materialize the DataFrame with timing
    let df = time_section!(stats, collect, { result_lf.collect()? });
    println!(
        "Collected DataFrame for get_numeric_stats_refactored, height: {}",
        df.height()
    );

    // Processing phase: compute the numeric stats from the DataFrame
    let numeric_stats = time_section!(stats, processing, {
        if df.height() == 0 {
            NumericStats {
                min: None,
                max: None,
                avg: None,
            }
        } else {
            let min_val = df.column("min")?.f64()?.get(0);
            let max_val = df.column("max")?.f64()?.get(0);
            let avg_val = df.column("avg")?.f64()?.get(0);
            NumericStats {
                min: min_val,
                max: max_val,
                avg: avg_val,
            }
        }
    });

    // Update overall timing and memory usage
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();
    stats.print_summary();

    log::info!(
        "get_numeric_stats_refactored for '{}' took {:?}",
        field_name,
        overall_start.elapsed()
    );
    Ok((numeric_stats, stats))
}

// --- Main Function (Unchanged from previous fix) ---
fn main() -> PolarsResult<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let num_records = 10_000_000;
    let parquet_file_path = "output_log_data_optimized_fixed.parquet"; // Changed filename
    let compression = ParquetCompression::Zstd(None);
    let row_group_size = Some(512 * 1024);
    let low_memory = std::env::args().any(|arg| arg == "--low-memory=true");

    log::info!("Generating {} records...", num_records);
    let gen_start = Instant::now();
    let base_time = Utc::now();
    let data: Vec<LogRecord> = (0..num_records)
        .into_par_iter()
        .map(|i| generate_random_log_record(i, base_time))
        .collect();
    log::info!(
        "Generated {} records in {:?}",
        data.len(),
        gen_start.elapsed()
    );

    log::info!(
        "Writing records to single Parquet file (Optimized & Fixed): {}",
        parquet_file_path
    );
    write_records_to_single_parquet_optimized(
        data,
        parquet_file_path,
        compression,
        row_group_size,
    )?;
    log::info!("Parquet writing complete.");

    log::info!("Pausing for 5 seconds before querying...");
    std::thread::sleep(StdDuration::from_secs(10));

    log::info!("Starting queries (low_memory = {})...", low_memory);

    let query_doc_ids: Vec<i64> = (0..100).map(|i| i * (num_records / 100) as i64).collect();
    let mut all_query_stats = Vec::new();

    match get_field_values_by_doc_ids_refactored(
        "level",
        &query_doc_ids,
        parquet_file_path,
        low_memory,
    ) {
        Ok((result, stats)) => {
            log::info!(
                "Result for get_field_values_by_doc_ids('level', specific_ids): {} unique levels found.",
                result.value_map.len()
            );
            all_query_stats.push(stats);
        }
        Err(e) => log::error!("Error querying field values by doc_ids (level): {}", e),
    }

    std::thread::sleep(StdDuration::from_secs(10));

    match get_field_values_by_doc_ids_refactored(
        "source_region",
        &query_doc_ids,
        parquet_file_path,
        low_memory,
    ) {
        Ok((result, stats)) => {
            log::info!(
                "Result for get_field_values_by_doc_ids('source_region', specific_ids): {} unique levels found.",
                result.value_map.len()
            );
            all_query_stats.push(stats);
        }
        Err(e) => log::error!("Error querying field values by doc_ids (region): {}", e),
    }

    std::thread::sleep(StdDuration::from_secs(10));

    match get_field_values_refactored("source_host", parquet_file_path, low_memory) {
        Ok((result, stats)) => {
            log::info!(
                "Result for get_field_values_refactored('source_host'): {} unique hosts found.",
                result.value_map.len()
            );
            all_query_stats.push(stats);
        }
        Err(e) => log::error!("Error querying field values (host): {}", e),
    }

    std::thread::sleep(StdDuration::from_secs(10));

    match get_numeric_stats_by_doc_ids_refactored(
        "payload_size",
        &query_doc_ids,
        parquet_file_path,
        low_memory,
    ) {
        Ok((result, stats)) => {
            log::info!(
                "Result for get_field_values_by_doc_ids('source_region', specific_ids): {:?} unique regions found.",
                result
            );
            all_query_stats.push(stats);
        }
        Err(e) => log::error!("Error querying numeric stats by doc_ids (payload): {}", e),
    }

    std::thread::sleep(StdDuration::from_secs(10));

    match get_numeric_stats_by_doc_ids_refactored(
        "user_metrics_login_time_ms",
        &query_doc_ids,
        parquet_file_path,
        low_memory,
    ) {
        Ok((result, stats)) => {
            log::info!(
                "Result for get_field_values_by_doc_ids('source_region', specific_ids): {:?} unique regions found.",
                result
            );
            all_query_stats.push(stats);
        }
        Err(e) => log::error!(
            "Error querying numeric stats by doc_ids (login_time): {}",
            e
        ),
    }

    std::thread::sleep(StdDuration::from_secs(10));

    match get_numeric_stats_refactored("user_metrics_clicks", parquet_file_path, low_memory) {
        Ok((result, stats)) => {
            log::info!(
                "Result for get_numeric_stats_refactored('user_metrics_clicks'): {:?}",
                result
            );
            all_query_stats.push(stats);
        }
        Err(e) => log::error!("Error querying numeric stats (clicks): {}", e),
    }

    log::info!("Query examples finished.");
    Ok(())
}
