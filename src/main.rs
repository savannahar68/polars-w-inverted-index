// src/main.rs
use chrono::Utc;
use polars::datatypes::IdxCa;
use polars::io::SerReader; // only SerReader is needed
use polars::prelude::*;
use rand::Rng;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};
use uuid::Uuid;

// jemalloc for approximate in-process memory stats
use jemalloc_ctl::{epoch, stats};
use jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn get_allocated_memory() -> Option<usize> {
    // advance the epoch so stats are up to date
    epoch::advance().ok()?;
    stats::allocated::read().ok() // unit struct, no ()
}

const NUM_RECORDS: usize = 10_000_000;
const JSONL_PATH: &str = "large_data_no_id.jsonl";
const PARQUET_PATH: &str = "large_data_no_id.parquet";
const NUM_INDICES_TO_FETCH: usize = 1_000;

/// Generate one nested JSON record (no doc_id field)
fn generate_random_json(i: usize) -> Value {
    let mut rng = rand::thread_rng();
    let levels = ["info", "warn", "error", "debug"];
    let regions = ["us-east-1", "eu-west-2", "ap-south-1"];
    json!({
        "timestamp": Utc::now().timestamp_millis(),
        "level": levels[rng.gen_range(0..levels.len())],
        "message": format!("Log message for record {}", i),
        "source": {
            "ip": format!("192.168.{}.{}", rng.gen_range(1..255), rng.gen_range(1..255)),
            "host": format!("server-{}.local", rng.gen_range(1..100)),
            "region": regions[rng.gen_range(0..regions.len())],
        },
        "user": {
            "id": format!("user_{}", rng.gen_range(1000..10000)),
            "session_id": Uuid::new_v4().to_string(),
            "metrics": {
                "login_time_ms": rng.gen_range(10..500),
                "clicks": rng.gen_range(0..50),
                "active": rng.gen_bool(0.8),
            }
        },
        "payload_size": rng.gen_range(100..10_240),
        "tags": (0..rng.gen_range(1..6))
            .map(|i| format!("tag_{}", i * rng.gen_range(1..10)))
            .collect::<Vec<_>>()
    })
}

fn create_jsonl_file() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating {} JSONL records...", NUM_RECORDS);
    let start = Instant::now();
    let f = File::create(JSONL_PATH)?;
    let mut w = BufWriter::new(f);
    for i in 0..NUM_RECORDS {
        let v = generate_random_json(i);
        serde_json::to_writer(&mut w, &v)?;
        w.write_all(b"\n")?;
        if i % 100_000 == 0 && i > 0 {
            println!("  …{} records", i);
        }
    }
    w.flush()?;
    println!("JSONL generation took {:?}", start.elapsed());
    Ok(())
}

fn create_parquet_from_jsonl() -> Result<(), Box<dyn std::error::Error>> {
    println!("Converting JSONL → Parquet…");
    let start = Instant::now();
    let mut df = JsonLineReader::from_path(JSONL_PATH)?
        .infer_schema_len(Some(1_000))
        .finish()?;
    println!("Schema: {:?}", df.schema());
    let f = File::create(PARQUET_PATH)?;
    ParquetWriter::new(f)
        .with_compression(ParquetCompression::Zstd(None))
        .with_statistics(true)
        .with_row_group_size(Some(131_072))
        .with_data_page_size(None) // correct builder method
        .finish(&mut df)?;
    println!("Parquet conversion took {:?}", start.elapsed());
    Ok(())
}

/// Reads only the rows in `row_indices_to_fetch`, projecting just "level",
/// by iterating row‑groups in the Parquet file.
async fn benchmark_efficient_read_aggregate() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Memory‑efficient Parquet read benchmark ---");

    // pick 1k random row indices (sorted)
    let mut rng = rand::thread_rng();
    let mut row_indices: Vec<u32> = (0..NUM_INDICES_TO_FETCH)
        .map(|_| rng.gen_range(0..NUM_RECORDS) as u32)
        .collect();
    row_indices.sort_unstable();

    // measure before
    let before = get_allocated_memory();
    println!("jemalloc allocated before read: {:?}", before);

    // metadata + schema
    let f_meta = File::open(PARQUET_PATH)?;
    let mut pr_meta = ParquetReader::new(f_meta);
    let file_meta = pr_meta.get_metadata()?.clone();
    let schema = pr_meta.schema()?;
    let proj_idx = schema
        .as_ref()
        .fields
        .iter()
        .position(|fld| fld.name == "level")
        .ok_or_else(|| PolarsError::SchemaFieldNotFound("level".into()))?;
    println!("Projecting 'level' at column index {}", proj_idx);

    // batched reader (async feature)
    let f_batched = File::open(PARQUET_PATH)?;
    let mut batched = ParquetReader::new(f_batched)
        .with_projection(Some(vec![proj_idx]))
        .batched(NUM_RECORDS)?;

    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut offset = 0u64;
    let mut idx_ptr = 0;
    let mut processed = 0;

    for (rg_i, rg_meta) in file_meta.row_groups.iter().enumerate() {
        if idx_ptr >= row_indices.len() {
            break;
        }
        let nrows = rg_meta.num_rows() as u64;
        let start = offset;
        let end = offset + nrows;
        offset = end;

        // collect the subset of our targets that fall in this RG
        let mut rel_idxs = Vec::new();
        while idx_ptr < row_indices.len() {
            let global = row_indices[idx_ptr] as u64;
            if global < start {
                idx_ptr += 1; // skip older
            } else if global < end {
                rel_idxs.push((global - start) as u32);
                idx_ptr += 1;
            } else {
                break;
            }
        }

        // always advance the reader by 1 batch
        let maybe_dfs = batched.next_batches(1).await?; // async call :contentReference[oaicite:1]{index=1}
        if let Some(mut dfs) = maybe_dfs {
            let df_chunk = dfs.pop().expect("1 DF per batch");
            if !rel_idxs.is_empty() {
                let s = &df_chunk.get_columns()[0];
                let idx_ca = IdxCa::new("idx", &rel_idxs);
                let taken = s.take(&idx_ca)?;
                if let Ok(str_ca) = taken.str() {
                    for opt in str_ca {
                        if let Some(val) = opt {
                            *counts.entry(val.to_string()).or_default() += 1;
                            processed += 1;
                        }
                    }
                }
            }
        }
    }

    // measure after
    let after = get_allocated_memory();
    println!("jemalloc allocated after read:  {:?}", after);
    if let (Some(b), Some(a)) = (before, after) {
        println!(
            "Approx Δ allocated: {:.2} MB",
            (a as f64 - b as f64) / 1024.0 / 1024.0
        );
    }
    println!("Processed {} rows, level counts:", processed);
    let mut kvs: Vec<_> = counts.into_iter().collect();
    kvs.sort_by_key(|(k, _)| k.clone());
    for (lvl, cnt) in kvs {
        println!("  {}: {}", lvl, cnt);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("INFO: using jemallocator for in-process stats");

    // if !Path::new(JSONL_PATH).exists() {
    //     create_jsonl_file()?;
    // } else {
    //     println!("Using existing JSONL: {}", JSONL_PATH);
    // }

    // if !Path::new(PARQUET_PATH).exists() {
    //     create_parquet_from_jsonl()?;
    // } else {
    //     println!("Using existing Parquet: {}", PARQUET_PATH);
    // }

    println!("Sleeping 20s before benchmark…");
    sleep(Duration::from_secs(20));

    benchmark_efficient_read_aggregate().await?;

    println!("Sleeping 20s before benchmark…");
    sleep(Duration::from_secs(20));

    println!("\nDone.");
    Ok(())
}
