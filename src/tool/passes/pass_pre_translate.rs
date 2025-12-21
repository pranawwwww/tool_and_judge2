use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use pyo3::pyfunction;
use serde::{Deserialize, Serialize};

use crate::{
    config::{Model, ToolConfig},
    tool::{
        base_path::{BASE_DATASET_PATH, BASE_RESULT_PATH},
        bfcl_formats::BfclDatasetEntry,
        experiments::{DatasetFileName, PreTranslateFileName, PreTranslateMode},
    },
    utils::{
        compare_id, get_model_safe_name, load_json_lines, model_name_to_safe_name, write_json_lines_to_file
    },
};

#[derive(Clone, Serialize)]
pub struct PreTranslateAggregatedInputQuestionEntry {
    pub id: String,
    pub question: String,
    pub file_name: DatasetFileName,
}
#[derive(Clone, Deserialize)]
pub struct PreTranslateAggregatedOutputQuestionEntry {
    pub id: String,
    pub original_question: String,
    pub translated_question: String,
    pub file_name: DatasetFileName,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct PreTranslateQuestionEntry{
    pub id: String,
    pub original_question: String,
    pub translated_question: String,
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
pub fn pass_pre_translation_aggregated_questions_output_file_path(config: &ToolConfig) -> String {
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
    let mut aggregated_entries: Vec<PreTranslateAggregatedInputQuestionEntry> = vec![];
    let mut dataset_file_names: HashSet<DatasetFileName> = HashSet::new();
    for experiment in config.experiments.iter() {
        let pre_translate_mode = PreTranslateMode::from_config_experiment(experiment);
        if let PreTranslateMode::PreTranslate = pre_translate_mode {
            let dataset_file_name = DatasetFileName::from_config_experiment(experiment);
            dataset_file_names.insert(dataset_file_name);
        }
    }
    for dataset_file_name in dataset_file_names.iter() {
        let dataset_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(dataset_file_name).unwrap()
        );
        let result_file_name_str = dataset_file_name_str.clone();
        let dataset_file_path = BASE_DATASET_PATH.join(&dataset_file_name_str);
        let result_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("pre_translate")
            .join(&result_file_name_str);
        let dataset_entries =
            load_json_lines(&dataset_file_path).expect("Unable to load dataset file");
        let dataset_entries_parsed: HashMap<String, BfclDatasetEntry> = dataset_entries
            .into_iter()
            .map(|entry_json| {
                let entry: BfclDatasetEntry =
                    serde_json::from_value(entry_json).expect("Unable to parse dataset entry");
                (entry.id.clone(), entry)
            })
            .collect();
        let ids: HashSet<String> = dataset_entries_parsed.keys().cloned().collect();
        let result_existing_ids: HashSet<String> = match load_json_lines(&result_file_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry_json| {
                    let entry: BfclDatasetEntry =
                        serde_json::from_value(entry_json).expect("Unable to parse result entry");
                    entry.id
                })
                .collect(),
            Err(_) => {
                println!(
                    "Result file {} does not exist, will be created.",
                    result_file_path.to_str().unwrap()
                );
                HashSet::new()
            }
        };
        let missing_ids: HashSet<String> = ids.difference(&result_existing_ids).cloned().collect();
        println!(
            "For dataset file {}, total entries: {}, existing entries: {}, missing entries: {}",
            dataset_file_name_str,
            ids.len(),
            result_existing_ids.len(),
            missing_ids.len(),
        );
        for missing_id in missing_ids.iter() {
            let entry = dataset_entries_parsed
                .get(missing_id)
                .expect("Missing ID should exist in dataset entries");
            let question_entry = PreTranslateAggregatedInputQuestionEntry {
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
    println!("Wrote aggregated questions to file: {}", output_file_path);
}

#[pyfunction]
pub fn pass_pre_translation_dispatch_results(config: &ToolConfig) {
    let output_path = pass_pre_translation_aggregated_questions_output_file_path(config);
    let output_entries = load_json_lines(&output_path)
        .expect("Unable to load pre-translation aggregated questions output file");
    let output_entries_parsed: HashMap<(String, DatasetFileName), PreTranslateAggregatedOutputQuestionEntry> = output_entries
        .into_iter()
        .map(|entry_json| {
            let entry: PreTranslateAggregatedOutputQuestionEntry =
                serde_json::from_value(entry_json).expect("Unable to parse question entry");
            ((entry.id.clone(), entry.file_name.clone()), entry)
        })
        .collect();

    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    for experiment in config.experiments.iter() {
        let pre_translate_mode = PreTranslateMode::from_config_experiment(experiment);
        let PreTranslateMode::PreTranslate = pre_translate_mode else {
            continue;
        };
        let dataset_file_name = DatasetFileName::from_config_experiment(experiment);
        let dataset_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&dataset_file_name).unwrap()
        );
        let result_file_name_str = dataset_file_name_str.clone();
        let dataset_file_path = BASE_DATASET_PATH.join(&dataset_file_name_str);
        let result_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("pre_translate")
            .join(&result_file_name_str);
        let dataset_entries = load_json_lines(&dataset_file_path)
            .expect("Unable to load dataset file for dispatching pre-translation results");
        let ids = dataset_entries
            .into_iter()
            .map(|entry| {
                let parsed_entry: BfclDatasetEntry =
                    serde_json::from_value(entry).expect("Unable to parse dataset entry");
                parsed_entry.id
            })
            .collect::<HashSet<String>>();
        let mut result_existing_entries: Vec<PreTranslateQuestionEntry> = match load_json_lines(&result_file_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry_json| {
                    serde_json::from_value(entry_json).expect("Unable to parse result entry")
                })
                .collect(),
            Err(_) => {
                println!(
                    "Result file {} does not exist, will be created.",
                    result_file_path.to_str().unwrap()
                );
                vec![]
            }
        };
        let result_existing_ids: HashSet<String> = result_existing_entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect();
        let result_missing_ids: HashSet<String> = ids.difference(&result_existing_ids).cloned().collect();
        let mut missing_count = 0;
        for missing_id in result_missing_ids.iter() {
            let key = (missing_id.clone(), dataset_file_name.clone());
            if let Some(output_entry) = output_entries_parsed.get(&key) {
                let question_entry = PreTranslateQuestionEntry {
                    id: output_entry.id.clone(),
                    original_question: output_entry.original_question.clone(),
                    translated_question: output_entry.translated_question.clone(),
                };
                result_existing_entries.push(question_entry);
            } else {
                missing_count += 1;
            }
        }
        println!(
            "For result file {}, target entries: {}, dispatched entries: {}, missing entries: {}",
            result_file_name_str, ids.len(), result_existing_entries.len(), missing_count
        );
        result_existing_entries.sort_by(|a, b| compare_id(&a.id, &b.id));
        let result_existing_entries_serialized: Vec<serde_json::Value> = result_existing_entries
            .into_iter()
            .map(|entry| serde_json::to_value(entry).unwrap())
            .collect();
        write_json_lines_to_file(&result_file_path, &result_existing_entries_serialized)
            .expect("Unable to write pre-translation result file");
        println!("Wrote pre-translation result file: {}", result_file_path.to_str().unwrap());
    }
    // remove the output file
    std::fs::remove_file(&output_path).expect("Unable to remove pre-translation aggregated questions output file");
    println!("Removed pre-translation aggregated questions output file: {}", output_path);
}
