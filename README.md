# Polars Document Benchmark

A high-performance log data processing system built with Rust and Polars for efficient querying of large log datasets.

## Performance Metrics

### Data Processing

- **8-9GB** of JSON data ingested
- **700MB** of Parquet data generated
- **500MB** memory footprint for running all queries (columns are not actively dropped)

### Query Performance

The system demonstrates excellent performance across various query types:

| Query Type                     | Field                      | Doc IDs | Result Rows | Total Time | Memory Impact |
| ------------------------------ | -------------------------- | ------- | ----------- | ---------- | ------------- |
| `get_field_values_by_doc_ids`  | level                      | 100     | 100         | 1.10s      | +272.47MB     |
| `get_field_values_by_doc_ids`  | source_region              | 100     | 100         | 1.01s      | +56.67MB      |
| `get_field_values_refactored`  | source_host                | -       | 10,000,000  | 1.14s      | +71.08MB      |
| `get_numeric_stats_by_doc_ids` | payload_size               | 100     | 1           | 1.01s      | +85.95MB      |
| `get_numeric_stats_by_doc_ids` | user_metrics_login_time_ms | 100     | 1           | 1.02s      | +6.56MB       |
| `get_numeric_stats_refactored` | user_metrics_clicks        | -       | 0           | 0.61s      | +23.47MB      |

## Query DSL

The system implements a high-level query DSL that allows for efficient querying of log data. The main query types are:

### Field Value Queries

1. **Get Field Values by Document IDs**

   ```rust
   get_field_values_by_doc_ids_refactored(field_name, doc_ids, file_path, low_memory)
   ```

   Retrieves specific field values for given document IDs.

2. **Get All Field Values**
   ```rust
   get_field_values_refactored(field_name, file_path, low_memory)
   ```
   Retrieves all values for a specified field across the dataset.

### Numeric Statistic Queries

1. **Get Numeric Stats by Document IDs**

   ```rust
   get_numeric_stats_by_doc_ids_refactored(field_name, doc_ids, file_path, low_memory)
   ```

   Calculates min, max, and average for a numeric field across specified document IDs.

2. **Get All Numeric Stats**
   ```rust
   get_numeric_stats_refactored(field_name, file_path, low_memory)
   ```
   Calculates min, max, and average for a numeric field across the entire dataset.

## Architecture

The system consists of the following components:

### Data Model

- **LogRecord**: Main data structure representing a log entry
- **LogSource**: Contains information about the source of the log
- **User**: Contains user-related information
- **UserMetrics**: Contains metrics about user activity
- **Answer**: Contains information about responses

### Key Optimizations

1. **Efficient DataFrame Creation**

   - Optimized column creation for different data types
   - Proper handling of complex types (lists, structs)

2. **Optimized Parquet Writing**

   - Categorical data type usage for string columns
   - Efficient sorting before writing
   - Compression and statistics generation

3. **Query Optimizations**
   - LazyFrame operations for delayed execution
   - Efficient filtering and joining
   - Memory-conscious processing

## Usage Example

```rust
// Generate sample data
let num_records = 10_000_000;
let data = (0..num_records)
    .into_par_iter()
    .map(|i| generate_random_log_record(i, base_time))
    .collect();

// Write to Parquet
write_records_to_single_parquet_optimized(
    data,
    "output_log_data.parquet",
    ParquetCompression::Zstd(None),
    Some(512 * 1024)
);

// Query examples
let query_doc_ids: Vec<i64> = (0..100).map(|i| i * (num_records / 100) as i64).collect();

// Get field values for specific docs
let (result, stats) = get_field_values_by_doc_ids_refactored(
    "level",
    &query_doc_ids,
    "output_log_data.parquet",
    false
);

// Get numeric stats across all docs
let (result, stats) = get_numeric_stats_refactored(
    "user_metrics_clicks",
    "output_log_data.parquet",
    false
);
```

## Performance Best Practices

1. **Categorical Data Types**

   - Use categorical columns for string fields with low cardinality
   - Significantly reduces memory usage and improves performance

2. **Efficient Filtering**

   - Create specific filters before executing joins
   - Use LazyFrame operations to defer computation

3. **Memory Management**

   - Low memory mode available for constrained environments
   - Manage column selection to reduce memory footprint

4. **Parallel Processing**
   - Leverage Rayon for parallel data generation
   - Efficient use of parallel processing for large datasets

## Query Timing Breakdown

Each query provides detailed timing information:

- **Setup**: Initial query preparation
- **Filter**: Filter creation time
- **Join**: Join operation time
- **Collect**: Time to materialize the LazyFrame
- **Processing**: Post-collection processing time

## Requirements

- Rust (latest stable)
- Dependencies:
  - polars
  - rayon
  - chrono
  - serde
  - uuid
  - rand
