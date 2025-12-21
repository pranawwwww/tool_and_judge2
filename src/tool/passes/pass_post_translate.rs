use core::panic;
use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyo3::pyfunction;

use crate::{
    config::ToolConfig,
    one_entry_map::KeyValuePair,
    tool::{
        base_path::BASE_RESULT_PATH,
        bfcl_formats::BfclOutputFunctionCall,
        experiments::{ParseOutputFileName, PostTranslateFileName, PostTranslateMode},
        passes::pass_parse_output::ParseOutputEntry,
    },
    utils::{compare_id, get_model_safe_name, load_json_lines, write_json_lines_to_file},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize)]
pub struct PostTranslateAggregatedInputEntry {
    pub parameter_value_to_translate: String,
}

#[derive(Clone, Deserialize)]
pub struct PostTranslateAggregatedOutputEntry {
    pub original_parameter_value: String,
    pub translated_parameter_value: String,
}

pub type PostTranslateEntry = ParseOutputEntry;

const POST_TRANSLATE_AGGREGATED_INPUT_FILE_NAME: &str = "post_translate_aggregated_input.jsonl";
const POST_TRANSLATE_AGGREGATED_OUTPUT_FILE_NAME: &str = "post_translate_aggregated_output.jsonl";

#[pyfunction]
pub fn pass_post_translate_aggregated_input_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(POST_TRANSLATE_AGGREGATED_INPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}
#[pyfunction]
pub fn pass_post_translate_aggregated_output_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(POST_TRANSLATE_AGGREGATED_OUTPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}

#[pyfunction]
pub fn pass_post_translate_prepare_aggregated_input(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    // deduplicate and select only experiments with post translation on
    let result_file_names: HashSet<PostTranslateFileName> = config
        .experiments
        .iter()
        .filter_map(|experiment| {
            let post_translate_mode = PostTranslateMode::from_config_experiment(experiment);
            match post_translate_mode {
                PostTranslateMode::PostTranslate => {
                    Some(PostTranslateFileName::from_config_experiment(experiment))
                }
                PostTranslateMode::NoPostTranslate => None,
            }
        })
        .collect();
    let mut parameter_values_to_translate: HashSet<String> = HashSet::new();
    for result_file_name in result_file_names.iter() {
        let result_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&result_file_name).unwrap()
        );
        let parse_output_file_name_str = result_file_name_str.clone();
        let parse_output_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("parse_output")
            .join(&parse_output_file_name_str);
        let result_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("post_translate")
            .join(&result_file_name_str);
        let parse_output_entries =
            load_json_lines(&parse_output_file_path).expect("Failed to load parse output file");
        let parse_output_entries_parsed: Vec<ParseOutputEntry> = parse_output_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry).expect("Parse output entry has wrong format")
            })
            .collect();
        let ids: HashSet<String> = parse_output_entries_parsed
            .iter()
            .map(|entry| entry.id.clone())
            .collect();
        let result_existing_ids: HashSet<String> = match load_json_lines(&result_file_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed_entry: PostTranslateEntry = serde_json::from_value(entry)
                        .expect("Post translate entry has wrong format");
                    parsed_entry.id
                })
                .collect(),
            Err(_) => {
                println!("Post translate result file not found, assuming no existing entries.");
                HashSet::new()
            }
        };
        let missing_ids: HashSet<String> = ids.difference(&result_existing_ids).cloned().collect();
        println!(
            "Post translate: for result file {:?}, found {} total entries, {} existing entries, {} missing entries.",
            result_file_name,
            ids.len(),
            result_existing_ids.len(),
            missing_ids.len()
        );
        for missing_id in missing_ids.iter() {
            let entry = parse_output_entries_parsed
                .iter()
                .find(|entry| &entry.id == missing_id)
                .expect("Missing ID not found in parse output entries");
            let Ok(parsed_function_calls) = &entry.result else {
                continue;
            };

            for function_call in parsed_function_calls.iter() {
                for (_param_name, param_value) in function_call.0.value.iter() {
                    let collected_values = collect_parameter_values(param_value);
                    parameter_values_to_translate.extend(collected_values);
                }
            }
        }
    }
    let aggregated_entries = parameter_values_to_translate
        .into_iter()
        .map(|param_value| PostTranslateAggregatedInputEntry {
            parameter_value_to_translate: param_value,
        })
        .collect::<Vec<PostTranslateAggregatedInputEntry>>();
    let aggregated_entries_serialized: Vec<serde_json::Value> = aggregated_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).unwrap())
        .collect();
    // Rust's output path is Python's input path
    let output_file_path = pass_post_translate_aggregated_input_file_path(config);
    write_json_lines_to_file(&output_file_path, &aggregated_entries_serialized)
        .expect("Failed to write post translate aggregated input entries to file");
    println!(
        "Pass post translate prepare aggregated input completed for model {}",
        model_safe_name
    );
}

#[pyfunction]
pub fn pass_post_translate_dispatch_results(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let aggregated_file_path = pass_post_translate_aggregated_output_file_path(config);
    let aggregated_entries = load_json_lines(&aggregated_file_path)
        .expect("Failed to load post translate aggregated input file");
    let translation_map: HashMap<String, String> = aggregated_entries
        .into_iter()
        .map(|entry| {
            let parsed_entry: PostTranslateAggregatedOutputEntry = serde_json::from_value(entry)
                .expect("Post translate aggregated input entry has wrong format");
            (
                parsed_entry.original_parameter_value,
                parsed_entry.translated_parameter_value,
            )
        })
        .collect();
    for experiment in config.experiments.iter() {
        let post_translate_mode = PostTranslateMode::from_config_experiment(experiment);
        let PostTranslateMode::PostTranslate = post_translate_mode else {
            continue;
        };
        let parse_output_file_name = ParseOutputFileName::from_config_experiment(experiment);
        let parse_output_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&parse_output_file_name).unwrap()
        );
        let post_translate_file_name_str = parse_output_file_name_str.clone();
        let parse_output_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("parse_output")
            .join(&parse_output_file_name_str);
        let post_translate_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("post_translate")
            .join(&post_translate_file_name_str);
        let parse_output_entries =
            load_json_lines(&parse_output_file_path).expect("Failed to load parse output file");
        let parse_output_entries_parsed: HashMap<String, ParseOutputEntry> = parse_output_entries
            .into_iter()
            .map(|entry| {
                let parsed_entry: ParseOutputEntry =
                    serde_json::from_value(entry).expect("Parse output entry has wrong format");
                (parsed_entry.id.clone(), parsed_entry)
            })
            .collect();
        let ids = parse_output_entries_parsed
            .keys()
            .cloned()
            .collect::<HashSet<String>>();
        let mut existing_entries: Vec<PostTranslateEntry> =
            match load_json_lines(&post_translate_file_path) {
                Ok(existing_entries) => existing_entries
                    .into_iter()
                    .map(|entry| {
                        let parsed_entry: PostTranslateEntry = serde_json::from_value(entry)
                            .expect("Post translate entry has wrong format");
                        parsed_entry
                    })
                    .collect(),
                Err(_) => {
                    println!("Post translate result file not found, starting fresh.");
                    vec![]
                }
            };
        let existing_ids: HashSet<String> = existing_entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect();
        let missing_ids: HashSet<String> = ids.difference(&existing_ids).cloned().collect();
        let mut missing_count = 0;
        for missing_id in missing_ids.iter() {
            let entry = parse_output_entries_parsed
                .get(missing_id)
                .expect("Missing ID not found in parse output entries");
            let Ok(function_calls) = &entry.result else {
                existing_entries.push(PostTranslateEntry {
                    id: entry.id.clone(),
                    valid: false,
                    result: entry.result.clone(),
                });
                continue;
            };
            let translate_function_calls =
                |function_calls: &Vec<BfclOutputFunctionCall>| -> Option<Vec<BfclOutputFunctionCall>> {
                    let mut translated_function_calls: Vec<BfclOutputFunctionCall> = vec![];
                    for function_call in function_calls.iter() {
                        let mut translated_params: IndexMap<String, serde_json::Value> =
                            IndexMap::new();
                        for (param_name, param_value) in function_call.0.value.iter() {
                            let translated_value =
                                translate_param_value(param_value, &translation_map)?;
                            translated_params.insert(param_name.clone(), translated_value);
                        }
                        translated_function_calls.push(BfclOutputFunctionCall(KeyValuePair{
                            key: function_call.0.key.clone(),
                            value: translated_params,
                        }));
                    }
                    Some(translated_function_calls)
                };
            if let Some(translated_function_calls) = translate_function_calls(function_calls) {
                existing_entries.push(PostTranslateEntry {
                    id: entry.id.clone(),
                    valid: true,
                    result: Ok(translated_function_calls),
                });
            } else {
                missing_count += 1;
            }
        }
        println!(
            "Post translate: for parse output file {:?}, translated {} missing entries, {} entries failed to translate.",
            parse_output_file_name,
            missing_ids.len() - missing_count,
            missing_count,
        );
        existing_entries.sort_by(|a, b| compare_id(&a.id, &b.id));
        let existing_entries_serialized: Vec<serde_json::Value> = existing_entries
            .into_iter()
            .map(|entry| serde_json::to_value(entry).unwrap())
            .collect();
        write_json_lines_to_file(&post_translate_file_path, &existing_entries_serialized)
            .expect("Failed to write post translate entries to file");
        println!(
            "Pass post translate dispatch results completed for model {}, file {:?}",
            model_safe_name, parse_output_file_name
        );
    }
    std::fs::remove_file(&aggregated_file_path)
        .expect("Failed to remove post translate aggregated output file");
    println!(
        "Removed temporary aggregated output file {:?}",
        aggregated_file_path
    );
}

fn collect_parameter_values(parameter_value: &serde_json::Value) -> HashSet<String> {
    match parameter_value {
        serde_json::Value::Array(array) => {
            let mut collected_values: HashSet<String> = HashSet::new();
            for item in array.iter() {
                let item_values = collect_parameter_values(item);
                collected_values.extend(item_values);
            }
            collected_values
        }
        serde_json::Value::Object(obj) => {
            let mut collected_values: HashSet<String> = HashSet::new();
            for (key, val) in obj.iter() {
                // collect both key and value
                collected_values.insert(key.clone());
                let val_values = collect_parameter_values(val);
                collected_values.extend(val_values);
            }
            collected_values
        }
        serde_json::Value::String(s) => {
            let mut collected_values: HashSet<String> = HashSet::new();
            collected_values.insert(s.clone());
            collected_values
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            HashSet::new()
        }
    }
}

fn translate_param_value(
    value: &serde_json::Value,
    translation_map: &HashMap<String, String>,
) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::Array(array) => {
            let mut translated_array: Vec<serde_json::Value> = Vec::new();
            for item in array.iter() {
                let translated_item = translate_param_value(item, translation_map)?;
                translated_array.push(translated_item);
            }
            Some(serde_json::Value::Array(translated_array))
        }
        serde_json::Value::Object(obj) => {
            let mut translated_obj: serde_json::Map<String, serde_json::Value> =
                serde_json::Map::new();
            for (key, val) in obj.iter() {
                // an awkward workaround to unify processing json value and string keys
                let translated_key = translate_param_value(
                    &serde_json::Value::String(key.clone()),
                    translation_map,
                )?;
                let translated_key_str = match translated_key {
                    serde_json::Value::String(s) => s,
                    _ => panic!("Translated key is not a string"),
                };
                let translated_value = translate_param_value(val, translation_map)?;
                translated_obj.insert(translated_key_str, translated_value);
            }
            Some(serde_json::Value::Object(translated_obj))
        }
        serde_json::Value::String(s) => match translation_map.get(s) {
            Some(translated_str) => Some(serde_json::Value::String(translated_str.clone())),
            None => None,
        },
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            Some(value.clone())
        }
    }
}
