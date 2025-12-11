use serde_json::{Value, value};
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::io::{BufRead, BufReader};
pub struct CategoryCache(pub HashMap<(String, Vec<String>), String>);

impl CategoryCache {
    pub fn load_or_create(cache_path: &str) -> Self {
        let mut category_cache = HashMap::new();
        let Ok(file) = File::open(cache_path) else {
            println!("Category cache file not found. Creating new cache.");
            return CategoryCache(HashMap::new());
        };
        println!("Loading category cache from {}...", cache_path);
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.expect("Reading line returns an error");
            assert!(!line.trim().is_empty());
            let item = serde_json::from_str::<Value>(&line)
                .expect("Failed to parse json line, the cache is corrupted");
            let item = item.as_array().expect("Cache line is not an array");
            assert_eq!(
                item.len(),
                2,
                "Cache line does not have exactly two elements"
            );
            let key_pair = item[0].as_array().expect("Cache key is not an array");
            let value = item[1]
                .as_str()
                .expect("Cache value is not a string")
                .to_string();
            let key_query = key_pair[0]
                .as_str()
                .expect("First element of cache key is not a string");
            let key_ground_truths = key_pair[1]
                .as_array()
                .expect("Second element of cache key is not an array");
            let key_ground_truths: Vec<String> = key_ground_truths
                .iter()
                .map(|v| {
                    v.as_str()
                        .expect("ground truth array element is not a string")
                        .to_string()
                })
                .collect();
            category_cache.insert((key_query.to_string(), key_ground_truths), value);
        }
        CategoryCache(category_cache)
    }
    pub fn save(self, cache_path: &str) {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(cache_path)
            .expect("Unable to open cache file for writing");
        for (key, value) in &self.0 {
            let cache_entry = serde_json::json!([[key.0, key.1], value]);
            let line =
                serde_json::to_string(&cache_entry).expect("Failed to serialize cache entry");
            writeln!(file, "{}", line).expect("Failed to write cache entry to file");
        }
    }
}

