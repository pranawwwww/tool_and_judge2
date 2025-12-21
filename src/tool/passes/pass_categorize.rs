use std::{
    collections::{HashMap, HashSet},
    fs::File,
};

use pyo3::pyfunction;
use serde::{Deserialize, Serialize};

use crate::{
    config::ToolConfig,
    tool::{
        base_path::BASE_RESULT_PATH,
        category_cache::CategoryCache,
        error_analysis::{EvaluationError, ToolErrorCategory},
        experiments::CategorizeFileName,
        passes::pass_evaluate::EvaluateEntry,
    },
    utils::{compare_id, get_model_safe_name, load_json_lines, write_json_lines_to_file},
};

#[derive(Clone, Serialize)]
pub struct CategorizeAggregatedInputEntry {
    pub actual_value: String,
    pub expected_values: Vec<String>,
}

#[derive(Clone, Deserialize)]
pub struct CategorizeAggregatedOutputEntry {
    pub error_category: ToolErrorCategory,
    pub actual_value: String,
    pub expected_values: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CategorizeEntry {
    pub id: String,
    pub error_category: ToolErrorCategory,
    pub error: EvaluationError,
}

const CATEGORIZE_AGGREGATED_INPUT_FILE_NAME: &str = "categorize_aggregated_input.jsonl";
const CATEGORIZE_AGGREGATED_OUTPUT_FILE_NAME: &str = "categorize_aggregated_output.jsonl";
const CATEGORY_CACHE_FILE_NAME: &str = "category_cache.jsonl";
const CATEGORY_CACHE_LOCK_FILE_NAME: &str = "category_cache.lock";
#[pyfunction]
pub fn pass_categorize_aggregated_input_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(CATEGORIZE_AGGREGATED_INPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}

#[pyfunction]
pub fn pass_categorize_aggregated_output_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(CATEGORIZE_AGGREGATED_OUTPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}

/// This function only generates entries that need to be categorized by an LLM.
#[pyfunction]
pub fn pass_categorize_prepare_aggregated_input(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let result_file_names: HashSet<CategorizeFileName> = config
        .experiments
        .iter()
        .map(|experiment| CategorizeFileName::from_config_experiment(experiment))
        .collect();
    // let mut aggregated_entries: Vec<CategorizeAggregatedInputEntry> = vec![];
    let category_cache_path = BASE_RESULT_PATH.join(CATEGORY_CACHE_FILE_NAME);
    let category_cache_lock_path = BASE_RESULT_PATH.join(CATEGORY_CACHE_LOCK_FILE_NAME);
    println!("Acquiring lock for category cache file...");
    let lock_file = File::create(category_cache_lock_path)
        .expect("Failed to create lock file for category cache");
    lock_file.lock().expect("Failed to lock the lock file");
    println!("Acquired lock for category cache file.");
    let category_cache = CategoryCache::load_or_create(&category_cache_path);
    let mut errors_to_categorize: HashSet<(String, Vec<String>)> = HashSet::new();
    for result_file_name in result_file_names.iter() {
        let evaluate_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&result_file_name).unwrap()
        );
        let evaluate_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("evaluate")
            .join(&evaluate_file_name_str);
        let evaluate_entries =
            load_json_lines(&evaluate_file_path).expect("Failed to load evaluate file");
        let evaluate_entries_parsed: HashMap<String, EvaluateEntry> = evaluate_entries
            .into_iter()
            .map(|entry| {
                let entry_parsed: EvaluateEntry =
                    serde_json::from_value(entry).expect("Failed to parse evaluate entry");
                (entry_parsed.id.clone(), entry_parsed)
            })
            .collect();
        let ids: HashSet<String> = evaluate_entries_parsed.keys().cloned().collect();
        
        // overwrite existing results
        for id in ids.iter() {
            let evaluate_entry = evaluate_entries_parsed
                .get(id)
                .expect("Missing evaluate entry for id");
            let Some(error) = &evaluate_entry.error else {
                continue;
            };
            // if !matches!(error, EvaluationError::InvalidParamValue { .. }) {
            //     continue;
            // }
            let EvaluationError::InvalidParamValue {
                param: _,
                actual_value,
                expected_values,
                decoded_output: _,
            } = error
            else {
                continue;
            };
            let actual_value =
                serde_json::to_string(actual_value).expect("Should serialize actual value");
            let expected_values: Vec<String> = expected_values
                .iter()
                .map(|v| serde_json::to_string(v).expect("Should serialize expected value"))
                .collect();
            let cache_key = (actual_value.clone(), expected_values.clone());
            // cache hit
            if category_cache.0.contains_key(&cache_key) {
                continue;
            }

            errors_to_categorize.insert((actual_value, expected_values));
        }
    }
    println!("Releasing lock for category cache file...");
    lock_file.unlock().expect("Failed to unlock the lock file");
    println!("Released lock for category cache file.");
    let aggregated_entries: Vec<CategorizeAggregatedInputEntry> = errors_to_categorize
        .into_iter()
        .map(
            |(actual_value, expected_values)| CategorizeAggregatedInputEntry {
                actual_value,
                expected_values,
            },
        )
        .collect();
    let aggregated_entries_serialized: Vec<serde_json::Value> = aggregated_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).unwrap())
        .collect();
    let output_file_path = pass_categorize_aggregated_input_file_path(config);
    write_json_lines_to_file(&output_file_path, &aggregated_entries_serialized)
        .expect("Failed to write categorize aggregated input file");
    println!(
        "Wrote categorize aggregated input file at {}",
        output_file_path
    );
}

#[pyfunction]
pub fn pass_categorize_dispatch_results(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let output_file_path = pass_categorize_aggregated_output_file_path(config);
    let output_entries = load_json_lines(&output_file_path)
        .expect("Failed to load categorize aggregated output file");
    // let output_entries_parsed: HashMap<
    //     (String, CategorizeFileName),
    //     CategorizeAggregatedOutputEntry,
    // > = output_entries
    //     .into_iter()
    //     .map(|entry| {
    //         let entry_parsed: CategorizeAggregatedOutputEntry = serde_json::from_value(entry)
    //             .expect("Failed to parse categorize aggregated output entry");
    //         (
    //             (entry_parsed.id.clone(), entry_parsed.file_name.clone()),
    //             entry_parsed,
    //         )
    //     })
    //     .collect();
    let output_error_categories: HashMap<(String, Vec<String>), ToolErrorCategory> = output_entries
        .into_iter()
        .map(|entry| {
            let entry_parsed: CategorizeAggregatedOutputEntry = serde_json::from_value(entry)
                .expect("Failed to parse categorize aggregated output entry");
            let key = (
                entry_parsed.actual_value.clone(),
                entry_parsed.expected_values.clone(),
            );
            (key, entry_parsed.error_category.clone())
        })
        .collect();
    let category_cache_path = BASE_RESULT_PATH.join(CATEGORY_CACHE_FILE_NAME);
    let category_cache_lock_path = BASE_RESULT_PATH.join(CATEGORY_CACHE_LOCK_FILE_NAME);
    println!("Acquiring lock for category cache file...");
    let lock_file = File::create(category_cache_lock_path)
        .expect("Failed to create lock file for category cache");
    lock_file.lock().expect("Failed to lock the lock file");
    println!("Acquired lock for category cache file.");
    let mut category_cache = CategoryCache::load_or_create(&category_cache_path);
    for experiment in config.experiments.iter() {
        let categorize_file_name = CategorizeFileName::from_config_experiment(experiment);
        let categorize_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&categorize_file_name).unwrap()
        );
        let evaluate_file_name_str = categorize_file_name_str.clone();
        let evaluate_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("evaluate")
            .join(&evaluate_file_name_str);
        let categorize_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("categorize")
            .join(&categorize_file_name_str);
        let evalutate_entries =
            load_json_lines(&evaluate_file_path).expect("Failed to load evaluate file");
        let evaluate_entries_parsed: HashMap<String, EvaluateEntry> = evalutate_entries
            .into_iter()
            .map(|entry| {
                let entry_parsed: EvaluateEntry =
                    serde_json::from_value(entry).expect("Failed to parse evaluate entry");
                (entry_parsed.id.clone(), entry_parsed)
            })
            .collect();
        let ids: HashSet<String> = evaluate_entries_parsed.keys().cloned().collect();
        // let mut existing_entries: Vec<CategorizeEntry> =
        //     match load_json_lines(&categorize_file_path) {
        //         Ok(existing_entries_json) => existing_entries_json
        //             .into_iter()
        //             .map(|entry| {
        //                 serde_json::from_value(entry)
        //                     .expect("Failed to parse existing categorize entry")
        //             })
        //             .collect(),
        //         Err(_) => vec![],
        //     };
        // let existing_ids: HashSet<String> = existing_entries
        //     .iter()
        //     .map(|entry| entry.id.clone())
        //     .collect();
        // let missing_ids: HashSet<String> = ids.difference(&existing_ids).cloned().collect();
        let mut new_entries: Vec<CategorizeEntry> = vec![];
        let mut missing_count = 0;
        let mut cache_hits = 0;
        let mut cache_misses = 0;
        // overwrite existing results
        for id in ids.iter() {
            let evaluate_entry = evaluate_entries_parsed
                .get(id)
                .expect("Missing evaluate entry for id");
            let Some(error) = &evaluate_entry.error else {
                continue;
            };
            let error_category = match error {
                EvaluationError::NoFunctionCallsFound { .. }
                | EvaluationError::JsonDecodeError { .. }
                | EvaluationError::ParsingError { .. } => ToolErrorCategory::SyntaxError,
                EvaluationError::InvalidEntryCount { .. }
                | EvaluationError::WrongFuncName { .. }
                | EvaluationError::MissingRequiredParam { .. }
                | EvaluationError::UnexpectedParam { .. } => ToolErrorCategory::MiscError,
                EvaluationError::InvalidParamValue {
                    param: _,
                    actual_value,
                    expected_values,
                    decoded_output: _,
                } => {
                    let actual_value_str =
                        serde_json::to_string(actual_value).expect("Should serialize actual value");
                    let expected_values_str: Vec<String> = expected_values
                        .iter()
                        .map(|v| serde_json::to_string(v).expect("Should serialize expected value"))
                        .collect();
                    let cache_key = (actual_value_str, expected_values_str);
                    if let Some(error_category) = category_cache.0.get(&cache_key) {
                        cache_hits += 1;
                        error_category.clone()
                    } else if let Some(error_category) = output_error_categories.get(&cache_key) {
                        let category = error_category.clone();
                        cache_misses += 1;
                        category_cache.0.insert(cache_key, category.clone());
                        category
                    } else {
                        missing_count += 1;
                        continue;
                    }
                }
            };
            let categorize_entry = CategorizeEntry {
                id: id.clone(),
                error_category,
                error: error.clone(),
            };
            new_entries.push(categorize_entry);
        }
        println!(
            "For categorize file {:?}, total ids: {}, missing ids after dispatch: {}",
            categorize_file_name,
            ids.len(),
            missing_count
        );
        println!("Cache hits: {}, Cache misses: {}", cache_hits, cache_misses);
        new_entries.sort_by(|a, b| compare_id(&a.id, &b.id));
        let new_entries_serialized: Vec<serde_json::Value> = new_entries
            .into_iter()
            .map(|entry| serde_json::to_value(entry).unwrap())
            .collect();
        write_json_lines_to_file(&categorize_file_path, &new_entries_serialized)
            .expect("Failed to write categorize file after dispatch");
        println!(
            "Wrote categorize file after dispatch at {:?}",
            categorize_file_path
        );
    }
    category_cache.save(&category_cache_path);
    println!("Releasing lock for category cache file...");
    lock_file.unlock().expect("Failed to unlock the lock file");
    println!("Released lock for category cache file.");
    // remove the aggregated output file after dispatch
    std::fs::remove_file(&output_file_path)
        .expect("Failed to remove categorize aggregated output file after dispatch");
    println!(
        "Removed categorize aggregated output file at {} after dispatch",
        output_file_path
    );
}
