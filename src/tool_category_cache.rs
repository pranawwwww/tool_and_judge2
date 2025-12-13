use serde::Deserialize;
use serde::Serialize;
use serde_json::{Value, value};
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::io::{BufRead, BufReader};

pub struct CategoryCache(pub HashMap<(String, Vec<String>), String>);

#[derive(Serialize, Deserialize, Clone)]
pub struct CategoryCacheEntry {
    pub query: String,
    pub ground_truths: Vec<String>,
    pub category: String,
}

impl CategoryCache {
    pub fn load_or_create(cache_path: &str) -> Self {
        let Ok(file) = File::open(cache_path) else {
            println!("Category cache file not found. Creating new cache.");
            return CategoryCache(HashMap::new());
        };
        println!("Loading category cache from {}...", cache_path);
        let reader = BufReader::new(file);
        let mut category_cache_entries: Vec<CategoryCacheEntry> = Vec::new();
        for line in reader.lines() {
            let line = line.expect("Reading line returns an error");
            if line.trim().is_empty() {
                continue;
            }
            let entry: CategoryCacheEntry =
                serde_json::from_str(&line).expect("Failed to parse category cache entry");
            category_cache_entries.push(entry);
        }
        let category_cache = category_cache_entries
            .into_iter()
            .map(|entry| ((entry.query, entry.ground_truths), entry.category))
            .collect::<HashMap<(String, Vec<String>), String>>();
        CategoryCache(category_cache)
    }
    pub fn save(&self, cache_path: &str) {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(cache_path)
            .expect("Unable to open cache file for writing");
        let category_entries: Vec<CategoryCacheEntry> = self
            .0
            .iter()
            .map(|((query, ground_truths), category)| CategoryCacheEntry {
                query: query.clone(),
                ground_truths: ground_truths.clone(),
                category: category.clone(),
            })
            .collect();
        for entry in category_entries {
            let entry = serde_json::to_string(&entry).expect("Failed to serialize cache entry to JSON value");
            writeln!(file, "{}", entry).expect("Failed to write cache entry to file");
        }
    }
}
