use chrono::{DateTime, Duration as ChronoDuration, Utc};
use polars::datatypes::DataType;
use polars::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::thread::sleep;
// Remove unused import
// use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Serialize, Deserialize)]
struct LogSource {
    ip: String,
    host: String,
    region: String,
}

#[derive(Serialize, Deserialize)]
struct UserMetrics {
    login_time_ms: i64,
    clicks: i64,
    active: bool,
}

#[derive(Serialize, Deserialize)]
struct User {
    id: String,
    session_id: String,
    metrics: UserMetrics,
}

#[derive(Serialize, Deserialize)]
struct Answer {
    #[serde(rename = "nxDomain")]
    nx_domain: bool,
    response_time_ms: i64,
}

// Main log record matching the JSON structure
#[derive(Serialize, Deserialize)]
struct LogRecord {
    doc_id: i64,
    timestamp: String, // Keeping as String; Polars can parse
    level: String,
    message: String,
    source: LogSource,
    user: User,
    payload_size: i64,
    tags: Vec<String>,
    answers: Vec<Answer>,
    processed: bool,
}

fn generate_random_log_record(i: usize, base_time: DateTime<Utc>) -> LogRecord {
    let mut rng = rand::thread_rng(); // Use thread_rng per call for simplicity in parallel map
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
    let timestamp = base_time + ChronoDuration::milliseconds(rng.gen_range(-30000..30000));
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

// Create DataFrame from a batch of records
fn create_dataframe_from_records(records: &[LogRecord]) -> PolarsResult<DataFrame> {
    let doc_ids: Vec<i64> = records.iter().map(|r| r.doc_id).collect();
    let timestamps: Vec<&str> = records.iter().map(|r| r.timestamp.as_str()).collect();
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
    // For tags and answers, you can collect them as JSON strings
    let tags: Vec<String> = records
        .iter()
        .map(|r| serde_json::to_string(&r.tags).unwrap_or_default())
        .collect();
    let answers: Vec<String> = records
        .iter()
        .map(|r| serde_json::to_string(&r.answers).unwrap_or_default())
        .collect();
    let processed: Vec<bool> = records.iter().map(|r| r.processed).collect();

    DataFrame::new(vec![
        Series::new("doc_id".into(), doc_ids),
        Series::new("timestamp".into(), timestamps),
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
        Series::new("tags".into(), tags),
        Series::new("answers".into(), answers),
        Series::new("processed".into(), processed),
    ])
}

fn write_records_to_single_parquet(
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
        "Writing {} records to single Parquet file {}...",
        records.len(),
        file_path
    );

    // Create a single DataFrame from all records in memory.
    let mut df = create_dataframe_from_records(&records)?;
    df = df.sort(["doc_id"], Default::default())?;

    let file = File::create(file_path)?;
    let buf_writer = BufWriter::new(file);

    // Configure the Parquet writer.
    let mut parquet_writer = ParquetWriter::new(buf_writer)
        .with_compression(compression)
        .with_statistics(StatisticsOptions::full());

    if let Some(rg_size) = row_group_size {
        parquet_writer = parquet_writer.with_row_group_size(Some(rg_size));
    }

    parquet_writer.finish(&mut df)?;
    let duration = start_time.elapsed();
    log::info!(
        "Successfully wrote {} records to {} in {:?}",
        records.len(),
        file_path,
        duration
    );
    Ok(())
}

// fn write_records_to_parquet_chunked(
//     records: Vec<LogRecord>,
//     file_path: &str,
//     chunk_size: usize,
//     compression: ParquetCompression,
//     row_group_size: Option<usize>,
// ) -> PolarsResult<()> {
//     if records.is_empty() {
//         return Ok(());
//     }
//     let start_time = Instant::now();
//     log::info!("Starting Parquet write to {}...", file_path);

//     let file = File::create(file_path)?;
//     let buf_writer = BufWriter::new(file);

//     println!("creating data frame from records");

//     // Create DataFrame from the first chunk
//     let first_chunk_slice = &records[0..chunk_size.min(records.len())];
//     let mut initial_df = create_dataframe_from_records(first_chunk_slice)?;

//     println!("done creating data frame from records");

//     let mut parquet_writer = ParquetWriter::new(buf_writer)
//         .with_compression(compression)
//         .with_statistics(StatisticsOptions::full());

//     if let Some(rg_size) = row_group_size {
//         parquet_writer = parquet_writer.with_row_group_size(Some(rg_size));
//     }

//     let mut total_written = 0;
//     // Write the first chunk
//     println!("Writing first chunk (size {})...", initial_df.height());
//     parquet_writer.finish(&mut initial_df)?;
//     total_written += initial_df.height();

//     // For remaining chunks, create new writers for each chunk
//     for chunk in records.chunks(chunk_size).skip(1) {
//         let file = File::options().append(true).open(file_path)?;
//         let buf_writer = BufWriter::new(file);
//         let mut parquet_writer = ParquetWriter::new(buf_writer)
//             .with_compression(compression)
//             .with_statistics(StatisticsOptions::full());

//         if let Some(rg_size) = row_group_size {
//             parquet_writer = parquet_writer.with_row_group_size(Some(rg_size));
//         }

//         println!("Writing chunk (size {})...", chunk.len());
//         let mut df_chunk = create_dataframe_from_records(chunk)?;
//         parquet_writer.finish(&mut df_chunk)?;
//         total_written += chunk.len();
//     }

//     let duration = start_time.elapsed();
//     log::info!(
//         "Successfully wrote {} records to {} in {:?}",
//         total_written,
//         file_path,
//         duration
//     );
//     Ok(())
// }

// Helper function for field names - replace . with _
fn field_name_to_column(field_name: &str) -> String {
    field_name.replace(".", "_")
}

#[derive(Debug)]
struct FieldValueResult {
    value_map: HashMap<String, Vec<i64>>,
}

fn get_field_values_by_doc_ids(
    field_name: &str,
    doc_ids: &[i64],
    file_path: &str,
    low_memory: bool,
) -> PolarsResult<FieldValueResult> {
    println!(
        "Querying get_field_values_by_doc_ids for field '{}' with {} doc_ids",
        field_name,
        doc_ids.len()
    );
    let start_time = Instant::now();
    let args = ScanArgsParquet {
        low_memory,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(file_path, args)?;

    // Convert field name to column name format (replace '.' with '_')
    let column_name = field_name_to_column(field_name);
    // Instead of wrapping a Series, create a constant vector:
    let doc_ids_vec = doc_ids.to_vec();

    // Use the constant literal for filtering
    let result_lf = lf
        .filter(col("doc_id").is_in(lit(Series::new("doc_ids_const".into(), &doc_ids_vec))))
        .group_by([col(&column_name)])
        .agg([col("doc_id").list().0.alias("doc_ids_list")])
        .select([col(&column_name), col("doc_ids_list")]);

    let df = result_lf.collect()?;
    println!(
        "Collected DataFrame for get_field_values_by_doc_ids, height: {}",
        df.height()
    );

    let field_col = df.column(&column_name)?;
    let ids_col = df.column("doc_ids_list")?;
    let mut value_map = HashMap::new();

    for i in 0..df.height() {
        if let Ok(key_value) = field_col.get(i) {
            let key_str = key_value.to_string();
            let clean_key = key_str.trim_matches('"').to_string();

            match ids_col.get(i) {
                Ok(AnyValue::List(list_series)) => {
                    if let Ok(i64_ca) = list_series.i64() {
                        let ids_vec: Vec<i64> = i64_ca.into_iter().flatten().collect();
                        value_map.insert(clean_key, ids_vec);
                    }
                }
                Ok(other) => {
                    log::warn!("Expected a list, but got: {:?}", other);
                }
                Err(e) => {
                    log::warn!("Error retrieving value: {:?}", e);
                }
            }
        }
    }

    log::info!(
        "get_field_values_by_doc_ids for '{}' took {:?}",
        field_name,
        start_time.elapsed()
    );
    Ok(FieldValueResult { value_map })
}

fn get_field_values(
    field_name: &str,
    file_path: &str,
    low_memory: bool,
) -> PolarsResult<FieldValueResult> {
    println!("Querying get_field_values for field '{}'", field_name);
    let start_time = Instant::now();
    let args = ScanArgsParquet {
        low_memory,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(file_path, args)?;

    // Convert field name to column name format
    let column_name = field_name_to_column(field_name);

    let result_lf = lf
        .select([col(&column_name), col("doc_id")])
        .group_by([col(&column_name)])
        .agg([col("doc_id").list().0])
        .select([col(&column_name), col("doc_id").alias("doc_ids_list")]);

    let df = result_lf.collect()?;
    println!(
        "Collected DataFrame for get_field_values, height: {}",
        df.height()
    );

    // Process similar to get_field_values_by_doc_ids
    let field_col = df.column(&column_name)?;
    let ids_col = df.column("doc_ids_list")?;
    let mut value_map = HashMap::new();

    for i in 0..df.height() {
        // Safe unwrapping of field value
        if let Ok(key_value) = field_col.get(i) {
            let key_str = key_value.to_string();
            let clean_key = key_str.trim_matches('"').to_string();

            // Extract the list values safely
            match ids_col.get(i) {
                Ok(AnyValue::List(list_series)) => {
                    // Extract i64 values from the list, flattening out any None values.
                    if let Ok(i64_ca) = list_series.i64() {
                        let ids_vec: Vec<i64> = i64_ca.into_iter().flatten().collect();
                        value_map.insert(clean_key, ids_vec);
                    } else {
                        log::warn!(
                            "Could not convert list series to i64 for key '{}'",
                            clean_key
                        );
                    }
                }
                Ok(other) => {
                    log::warn!("Expected a list, but got: {:?}", other);
                }
                Err(e) => {
                    log::warn!("Error retrieving value: {:?}", e);
                }
            }
        }
    }

    log::info!(
        "get_field_values for '{}' took {:?}",
        field_name,
        start_time.elapsed()
    );
    Ok(FieldValueResult { value_map })
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumericStats {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub avg: Option<f64>,
}

fn get_numeric_stats_by_doc_ids(
    field_name: &str,
    doc_ids: &[i64],
    file_path: &str,
    low_memory: bool,
) -> PolarsResult<NumericStats> {
    println!(
        "Querying get_numeric_stats_by_doc_ids for field '{}' with {} doc_ids",
        field_name,
        doc_ids.len()
    );
    let start_time = Instant::now();
    let args = ScanArgsParquet {
        low_memory,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(file_path, args)?;

    // Convert field name to column name format
    let column_name = field_name_to_column(field_name);
    let doc_ids_series = Series::new("doc_id_filter".into(), doc_ids);
    let numeric_expr = col(&column_name).cast(DataType::Float64);

    let result_lf = lf.filter(col("doc_id").is_in(lit(doc_ids_series))).select([
        numeric_expr.clone().min().alias("min"),
        numeric_expr.clone().max().alias("max"),
        numeric_expr.mean().alias("avg"),
    ]);

    let df = result_lf.collect()?;
    println!(
        "Collected DataFrame for get_numeric_stats_by_doc_ids, height: {}",
        df.height()
    );

    if df.height() == 0 {
        return Ok(NumericStats {
            min: None,
            max: None,
            avg: None,
        });
    }

    // Extract values safely
    let min_val = df
        .column("min")?
        .get(0)
        .and_then(|av| av.try_extract::<f64>())
        .ok();
    let max_val = df
        .column("max")?
        .get(0)
        .and_then(|av| av.try_extract::<f64>())
        .ok();
    let avg_val = df
        .column("avg")?
        .get(0)
        .and_then(|av| av.try_extract::<f64>())
        .ok();

    let stats = NumericStats {
        min: min_val,
        max: max_val,
        avg: avg_val,
    };
    log::info!(
        "get_numeric_stats_by_doc_ids for '{}' took {:?}",
        field_name,
        start_time.elapsed()
    );
    Ok(stats)
}

fn get_numeric_stats(
    field_name: &str,
    file_path: &str,
    low_memory: bool,
) -> PolarsResult<NumericStats> {
    println!("Querying get_numeric_stats for field '{}'", field_name);
    let start_time = Instant::now();
    let args = ScanArgsParquet {
        low_memory,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(file_path, args)?;

    // Convert field name to column name format
    let column_name = field_name_to_column(field_name);
    let numeric_expr = col(&column_name).cast(DataType::Float64);

    let result_lf = lf.select([
        numeric_expr.clone().min().alias("min"),
        numeric_expr.clone().max().alias("max"),
        numeric_expr.mean().alias("avg"),
    ]);

    let df = result_lf.collect()?;
    println!(
        "Collected DataFrame for get_numeric_stats, height: {}",
        df.height()
    );

    if df.height() == 0 {
        return Ok(NumericStats {
            min: None,
            max: None,
            avg: None,
        });
    }

    // Extract values safely
    let min_val = df
        .column("min")?
        .get(0)
        .and_then(|av| av.try_extract::<f64>())
        .ok();
    let max_val = df
        .column("max")?
        .get(0)
        .and_then(|av| av.try_extract::<f64>())
        .ok();
    let avg_val = df
        .column("avg")?
        .get(0)
        .and_then(|av| av.try_extract::<f64>())
        .ok();

    let stats = NumericStats {
        min: min_val,
        max: max_val,
        avg: avg_val,
    };
    log::info!(
        "get_numeric_stats for '{}' took {:?}",
        field_name,
        start_time.elapsed()
    );
    Ok(stats)
}

fn main() -> PolarsResult<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    // --- Configuration ---
    let num_records = 10_000_000;
    let parquet_file_path = "output_log_data.parquet";
    let compression = ParquetCompression::Zstd(None); // Zstd default level
    let row_group_size = Some(512000); // 128 MiB
    let low_memory = true; // Memory-efficient mode

    // --- Phase 1: Data Generation ---
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

    // --- Phase 2: Parquet Writing ---
    log::info!(
        "Writing records to single Parquet file: {}",
        parquet_file_path
    );
    write_records_to_single_parquet(
        data, // data is moved here
        parquet_file_path,
        compression,
        row_group_size,
    )?;
    log::info!("Parquet writing complete. Original data vector dropped.");

    sleep(Duration::from_secs(30));

    // --- Phase 4: Querying ---
    log::info!("Starting queries (low_memory = {})...", low_memory);

    // Example Query 1: Get values for 'level' field for specific doc_ids
    let query_doc_ids: Vec<i64> = (0..100).map(|i| i * 1000).collect(); // Sample 100 doc ids
    match get_field_values_by_doc_ids("level", &query_doc_ids, parquet_file_path, low_memory) {
        Ok(result) => log::info!("Result for get_field_values_by_doc_ids('level', specific_ids): {} unique levels found.", result.value_map.len()),
        Err(e) => log::error!("Error querying field values by doc_ids: {}", e),
    }
    match get_field_values_by_doc_ids("source_region", &query_doc_ids, parquet_file_path, low_memory) {
        Ok(result) => log::info!("Result for get_field_values_by_doc_ids('source_region', specific_ids): {} unique regions found.", result.value_map.len()),
        Err(e) => log::error!("Error querying nested field values by doc_ids: {}", e),
    }

    // Example Query 2: Get all values for 'source_host' field
    match get_field_values("source_host", parquet_file_path, low_memory) {
        Ok(result) => log::info!(
            "Result for get_field_values('source_host'): {} unique hosts found.",
            result.value_map.len()
        ),
        Err(e) => log::error!("Error querying field values: {}", e),
    }

    // Example Query 3: Get numeric stats for 'payload_size' for specific doc_ids
    match get_numeric_stats_by_doc_ids(
        "payload_size",
        &query_doc_ids,
        parquet_file_path,
        low_memory,
    ) {
        Ok(stats) => log::info!(
            "Result for get_numeric_stats_by_doc_ids('payload_size', specific_ids): {:?}",
            stats
        ),
        Err(e) => log::error!("Error querying numeric stats by doc_ids: {}", e),
    }
    match get_numeric_stats_by_doc_ids("user_metrics_login_time_ms", &query_doc_ids, parquet_file_path, low_memory) {
        Ok(stats) => log::info!("Result for get_numeric_stats_by_doc_ids('user_metrics_login_time_ms', specific_ids): {:?}", stats),
        Err(e) => log::error!("Error querying nested numeric stats by doc_ids: {}", e),
    }

    // Example Query 4: Get overall numeric stats for 'user_metrics_clicks'
    match get_numeric_stats("user_metrics_clicks", parquet_file_path, low_memory) {
        Ok(stats) => log::info!(
            "Result for get_numeric_stats('user_metrics_clicks'): {:?}",
            stats
        ),
        Err(e) => log::error!("Error querying numeric stats: {}", e),
    }

    log::info!("Query examples finished.");
    Ok(())
}
