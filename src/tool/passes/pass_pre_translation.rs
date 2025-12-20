use std::{collections::{HashMap, HashSet}, path::PathBuf};

use pyo3::pyfunction;
use serde::Serialize;

use crate::{config::{Model, ToolConfig}, tool::{base_path::{BASE_DATASET_PATH, BASE_RESULT_PATH}, bfcl_formats::BfclDatasetEntry, experiments::{DatasetFileName, PreTranslateMode}}, utils::{get_model_safe_name, load_json_lines, model_name_to_safe_name, write_json_lines_to_file}};

#[derive(Clone, Serialize)]
pub struct QuestionEntry {
    pub id: String,
    pub question: String,
    pub file_name: DatasetFileName,
}
const PRE_TRANSLATION_AGGREGATED_QUESTIONS_INPUT_FILE_NAME: &str =
    "pre_translation_aggregated_questions_input.jsonl";
const PRE_TRANSLATION_AGGREGATED_QUESTIONS_OUTPUT_FILE_NAME: &str =
    "pre_translation_aggregated_questions_output.jsonl";
#[pyfunction]
pub fn pass_pre_translation_aggregated_questions_input_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(model_safe_name)
        .join(PRE_TRANSLATION_AGGREGATED_QUESTIONS_INPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}
#[pyfunction]
pub fn pass_pre_translation_aggregated_questions_output_file_path(
    config: &ToolConfig,
) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(model_safe_name)
        .join(PRE_TRANSLATION_AGGREGATED_QUESTIONS_OUTPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}

#[pyfunction]
pub fn pass_pre_translation_prepare_aggregated_questions(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let mut aggregated_entries: Vec<QuestionEntry> = vec![];
    let mut dataset_file_names: HashSet<DatasetFileName> = HashSet::new();
    for experiment in config.experiments.iter() {
        let pre_translate_mode = PreTranslateMode::from_config_experiment(experiment);
        if let PreTranslateMode::PreTranslate = pre_translate_mode {
            let dataset_file_name = DatasetFileName::from_config_experiment(experiment);
            dataset_file_names.insert(dataset_file_name);
        }
    }
    for dataset_file_name in dataset_file_names.iter() {
        let dataset_file_name_str = format!("{}.jsonl", serde_json::to_string(dataset_file_name).unwrap());
        let result_file_name_str = dataset_file_name_str.clone();
        let dataset_file_path = BASE_DATASET_PATH
            .join(&dataset_file_name_str);
        let result_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("pre_translate")
            .join(&result_file_name_str);
        let dataset_entries = load_json_lines(&dataset_file_path).expect("Unable to load dataset file");
        let dataset_entries_parsed: HashMap<String, BfclDatasetEntry> = dataset_entries
            .into_iter()
            .map(|entry_json| {
                let entry: BfclDatasetEntry = serde_json::from_value(entry_json).expect("Unable to parse dataset entry");
                (entry.id.clone(), entry)
            })
            .collect();
        let ids: HashSet<String> = dataset_entries_parsed.keys().cloned().collect();
        let result_existing_ids: HashSet<String> = match load_json_lines(&result_file_path) {
            Ok(entries) => {
                entries.into_iter()
                    .map(|entry_json| {
                        let entry: BfclDatasetEntry = serde_json::from_value(entry_json).expect("Unable to parse result entry");
                        entry.id
                    })
                    .collect()
            }
            Err(_) => {
                println!("Result file {} does not exist, will be created.", result_file_path.to_str().unwrap());
                HashSet::new()
            }
        };
        let missing_ids: HashSet<String> = ids.difference(&result_existing_ids).cloned().collect();
        println!("For dataset file {}, total entries: {}, existing entries: {}, missing entries: {}",
            dataset_file_name_str,
            ids.len(),
            result_existing_ids.len(),
            missing_ids.len(),
        );
        for missing_id in missing_ids.iter() {
            let entry = dataset_entries_parsed.get(missing_id).expect("Missing ID should exist in dataset entries");
            let question_entry = QuestionEntry {
                id: entry.id.clone(),
                question: entry.question[0][0].content.clone(),
                file_name: dataset_file_name.clone(),
            };
            aggregated_entries.push(question_entry);
        }
    }
    let aggregated_entries_serialized: Vec<serde_json::Value> = aggregated_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).unwrap())
        .collect();
    // Rust's output path is Python's input path
    let output_file_path = pass_pre_translation_aggregated_questions_input_file_path(config);
    write_json_lines_to_file(&output_file_path, &aggregated_entries_serialized)
        .expect("Unable to write aggregated questions to file");
    println!(
        "Wrote aggregated questions to file: {}",
        output_file_path
    );
}
