use std::collections::{HashMap, HashSet};

use pyo3::pyfunction;
use serde::{Deserialize, Serialize};

use crate::{
    config::{ToolConfig, ToolExperiment, TranslateMode},
    models::{
        function_name_mapper::{FunctionNameMapper},
        model_interface::get_model_interface,
    },
    tool::{
        base_path::{BASE_DATASET_PATH, BASE_RESULT_PATH},
        bfcl_formats::BfclDatasetEntry,
        experiments::{
            DatasetFileName, GenerateRawFileName, PreTranslateMode, PromptTranslateMode,
        },
        passes::pass_pre_translate::PreTranslateQuestionEntry,
    },
    utils::{compare_id, get_model_safe_name, load_json_lines, write_json_lines_to_file},
};

#[derive(Clone, Serialize)]
pub struct GenerateRawAggregatedInputEntry {
    pub id: String,
    pub question: String,
    pub tools: serde_json::Value,
    pub prompt_passing_in_english: bool,
    pub file_name: GenerateRawFileName,
}
#[derive(Clone, Deserialize)]
pub struct GenerateRawAggregatedOutputEntry {
    pub id: String,
    pub raw_output: String,
    pub file_name: GenerateRawFileName,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct GenerateRawEntry {
    pub id: String,
    pub raw_output: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FunctionNameMapperEntry {
    pub original_function_name: String,
    pub sanitized_function_name: String,
}
const GENERATE_RAW_AGGREGATED_INPUT_FILE_NAME: &str = "generate_raw_aggregated_input.jsonl";
const GENERATE_RAW_AGGREGATED_OUTPUT_FILE_NAME: &str = "generate_raw_aggregated_output.jsonl";

#[pyfunction]
pub fn pass_generate_raw_aggregated_input_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(GENERATE_RAW_AGGREGATED_INPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}
#[pyfunction]
pub fn pass_generate_raw_aggregated_output_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(GENERATE_RAW_AGGREGATED_OUTPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}
#[pyfunction]
pub fn pass_generate_raw_prepare_aggregated_input(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let model_interface = get_model_interface(model);
    let function_name_mapper = get_function_name_mapper();

    // deduplicate experiments on this pass
    let result_file_names: HashSet<GenerateRawFileName> = config
        .experiments
        .iter()
        .map(|experiment| GenerateRawFileName::from_config_experiment(experiment))
        .collect();
    let mut aggregated_entries: Vec<GenerateRawAggregatedInputEntry> = vec![];
    for result_file_name in result_file_names.iter() {
        // first assume no pre translation
        // let dataset_file_name = DatasetFileName::from_config_experiment(experiment);
        // let result_file_name = GenerateRawFileName::from_config_experiment(experiment);
        let dataset_file_name = DatasetFileName(
            result_file_name.0,
            result_file_name.1,
            result_file_name.2,
            result_file_name.4,
        );
        let dataset_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&dataset_file_name).unwrap()
        );
        let result_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&result_file_name).unwrap()
        );
        // it is Some if pre translation is enabled, otherwise None
        let translated_question_entries: Option<HashMap<String, PreTranslateQuestionEntry>> =
            match result_file_name.3 {
                PreTranslateMode::PreTranslate => {
                    let pre_translate_result_file_name_str = format!(
                        "{}.jsonl",
                        serde_json::to_string(&dataset_file_name).unwrap()
                    );
                    let pre_translate_result_file_path = BASE_RESULT_PATH
                        .join(&model_safe_name)
                        .join("pre_translate")
                        .join(&pre_translate_result_file_name_str);
                    let pre_translate_entries = load_json_lines(&pre_translate_result_file_path)
                        .expect("Failed to open pre-translate result file");
                    let pre_translate_entries_parsed: HashMap<String, PreTranslateQuestionEntry> =
                        pre_translate_entries
                            .into_iter()
                            .map(|entry| {
                                let parsed_entry: PreTranslateQuestionEntry =
                                    serde_json::from_value(entry)
                                        .expect("Pre-translate entry has wrong format");
                                (parsed_entry.id.clone(), parsed_entry)
                            })
                            .collect();
                    Some(pre_translate_entries_parsed)
                }
                PreTranslateMode::NoPreTranslate => None,
            };
        let dataset_file_path = BASE_DATASET_PATH.join(&dataset_file_name_str);
        let result_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("generate_raw")
            .join(&result_file_name_str);
        let dataset_entries = load_json_lines(&dataset_file_path).expect("Failed to open dataset");
        let dataset_entries_parsed: HashMap<String, BfclDatasetEntry> = dataset_entries
            .into_iter()
            .map(|entry| {
                let parsed_entry: BfclDatasetEntry =
                    serde_json::from_value(entry).expect("Dataset entry has wrong format");
                (parsed_entry.id.clone(), parsed_entry)
            })
            .collect();
        let ids: HashSet<String> = dataset_entries_parsed.keys().cloned().collect();
        let result_existing_ids: HashSet<String> = match load_json_lines(&result_file_path) {
            Ok(existing_results) => existing_results
                .into_iter()
                .map(|entry| {
                    let parsed_entry: GenerateRawEntry =
                        serde_json::from_value(entry).expect("Result entry has wrong format");
                    parsed_entry.id
                })
                .collect(),
            Err(_) => {
                println!(
                    "Result file {} does not exist, it will be created.",
                    result_file_path.to_str().unwrap()
                );
                HashSet::new()
            }
        };
        let missing_ids: Vec<String> = ids.difference(&result_existing_ids).cloned().collect();
        println!(
            "For result file {}, total entries: {}, existing entries: {}, missing entries to generate: {}",
            result_file_path.to_str().unwrap(),
            ids.len(),
            result_existing_ids.len(),
            missing_ids.len(),
        );
        let prompt_passing_in_english = match result_file_name.5 {
            PromptTranslateMode::PromptTranslate => true,
            PromptTranslateMode::NoPromptTranslate => false,
        };
        for id in missing_ids.iter() {
            let dataset_entry = dataset_entries_parsed
                .get(id)
                .expect("Missing ID should exist in dataset entries");
            let question = match &translated_question_entries {
                Some(translated_map) => {
                    let translated_entry = translated_map
                        .get(id)
                        .expect(&format!("Missing translated entry for ID {}", id));
                    translated_entry.translated_question.clone()
                }
                None => dataset_entry.question[0][0].content.clone(),
            };
            let tools = model_interface
                .generate_tool_definitions(&dataset_entry.function, &function_name_mapper);
            let input_entry = GenerateRawAggregatedInputEntry {
                id: dataset_entry.id.clone(),
                question,
                tools,
                prompt_passing_in_english,
                file_name: result_file_name.clone(),
            };
            aggregated_entries.push(input_entry);
        }
    }
    let aggregated_entries_serialized: Vec<serde_json::Value> = aggregated_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).unwrap())
        .collect();
    let aggregated_input_file_path = pass_generate_raw_aggregated_input_file_path(config);
    write_json_lines_to_file(&aggregated_input_file_path, &aggregated_entries_serialized)
        .expect("Failed to write aggregated input file");
    println!(
        "Wrote aggregated input file to {}",
        aggregated_input_file_path
    );
}

#[pyfunction]
pub fn pass_generate_raw_dispatch_results(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let output_file_path = pass_generate_raw_aggregated_output_file_path(config);
    let output_entries =
        load_json_lines(&output_file_path).expect("Failed to open aggregated output file");
    let output_entries_parsed: HashMap<
        (String, GenerateRawFileName),
        GenerateRawAggregatedOutputEntry,
    > = output_entries
        .into_iter()
        .map(|entry| {
            let parsed_entry: GenerateRawAggregatedOutputEntry =
                serde_json::from_value(entry).expect("Aggregated output entry has wrong format");
            (
                (parsed_entry.id.clone(), parsed_entry.file_name.clone()),
                parsed_entry,
            )
        })
        .collect();
    for experiment in config.experiments.iter() {
        let dataset_file_name = DatasetFileName::from_config_experiment(experiment);
        let dataset_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&dataset_file_name).unwrap()
        );
        let result_file_name = GenerateRawFileName::from_config_experiment(experiment);
        let result_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&result_file_name).unwrap()
        );
        let dataset_file_path = BASE_DATASET_PATH.join(&dataset_file_name_str);
        let result_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("generate_raw")
            .join(&result_file_name_str);
        let dataset_entries = load_json_lines(&dataset_file_path).expect("Failed to open dataset");
        let ids: HashSet<String> = dataset_entries
            .into_iter()
            .map(|entry| {
                let parsed_entry: BfclDatasetEntry =
                    serde_json::from_value(entry).expect("Dataset entry has wrong format");
                parsed_entry.id
            })
            .collect();
        let mut result_existing_entries: Vec<GenerateRawEntry> =
            match load_json_lines(&result_file_path) {
                Ok(existing_results) => existing_results
                    .into_iter()
                    .map(|entry| {
                        serde_json::from_value(entry).expect("Result entry has wrong format")
                    })
                    .collect(),
                Err(_) => {
                    println!(
                        "Result file {} does not exist, it will be created.",
                        result_file_path.to_str().unwrap()
                    );
                    vec![]
                }
            };
        let result_existing_ids: HashSet<String> = result_existing_entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect();
        let result_missing_ids: HashSet<String> =
            ids.difference(&result_existing_ids).cloned().collect();
        let mut missing_count = 0;
        for id in result_missing_ids.iter() {
            let key = (id.clone(), result_file_name.clone());
            if let Some(output_entry) = output_entries_parsed.get(&key) {
                let result_entry = GenerateRawEntry {
                    id: output_entry.id.clone(),
                    raw_output: output_entry.raw_output.clone(),
                };
                result_existing_entries.push(result_entry);
            } else {
                missing_count += 1;
            }
        }
        println!(
            "For result file {}, target entries: {}, dispatched entries: {}, missing entries: {}",
            result_file_name_str,
            ids.len(),
            result_existing_entries.len(),
            missing_count
        );
        result_existing_entries.sort_by(|a, b| compare_id(&a.id, &b.id));
        let result_entries_serialized: Vec<serde_json::Value> = result_existing_entries
            .into_iter()
            .map(|entry| serde_json::to_value(entry).unwrap())
            .collect();
        write_json_lines_to_file(&result_file_path, &result_entries_serialized)
            .expect("Failed to write result file");
        println!(
            "Wrote result file to {}",
            result_file_path.to_str().unwrap()
        );
    }
    // remove the aggregated output file after dispatching
    std::fs::remove_file(&output_file_path).expect("Failed to remove aggregated output file");
    println!("Removed aggregated output file {}", output_file_path);
}

pub fn generate_function_name_mapper_file() {
    let output_file_path = BASE_DATASET_PATH.join("function_name_mapper.jsonl");
    let vanilla_experiment = ToolExperiment {
        translate_mode: TranslateMode::NotTranslated {},
        add_noise_mode: crate::config::AddNoiseMode::NoNoise,
    };
    let dataset_file_name = DatasetFileName::from_config_experiment(&vanilla_experiment);
    let dataset_file_name_str = format!(
        "{}.jsonl",
        serde_json::to_string(&dataset_file_name).unwrap()
    );
    let dataset_path = BASE_DATASET_PATH.join(&dataset_file_name_str);
    let dataset_entries = load_json_lines(&dataset_path).expect("Failed to open dataset");
    let dataset_entries_parsed: Vec<BfclDatasetEntry> = dataset_entries
        .into_iter()
        .map(|entry| serde_json::from_value(entry).expect("Dataset entry has wrong format"))
        .collect();
    let mut original_to_sanitized: HashMap<String, String> = HashMap::new();
    let mut sanitized_names: HashSet<String> = HashSet::new();
    for entry in dataset_entries_parsed.iter() {
        for function in entry.function.iter() {
            let original_name = &function.name;
            if !original_to_sanitized.contains_key(original_name) {
                // Create sanitized name
                let mut sanitized = original_name.replace(".", "_");
                sanitized = sanitized
                    .chars()
                    .map(|c| {
                        if c.is_alphanumeric() || c == '_' || c == '-' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                if sanitized_names.contains(&sanitized) {
                    let mut counter = 1;
                    let base_sanitized = sanitized.clone();
                    while sanitized_names.contains(&format!("{}_{}", base_sanitized, counter)) {
                        counter += 1;
                    }
                    sanitized = format!("{}_{}", base_sanitized, counter);
                }
                original_to_sanitized.insert(original_name.clone(), sanitized.clone());
                sanitized_names.insert(sanitized);
            }
        }
    }
    let function_name_mapper_entries: Vec<FunctionNameMapperEntry> = original_to_sanitized
        .into_iter()
        .map(
            |(original_function_name, sanitized_function_name)| FunctionNameMapperEntry {
                original_function_name,
                sanitized_function_name,
            },
        )
        .collect();
    let serialized_entries: Vec<serde_json::Value> = function_name_mapper_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).unwrap())
        .collect();
    write_json_lines_to_file(&output_file_path, &serialized_entries)
        .expect("Failed to write function name mapper file");
    println!(
        "Wrote function name mapper file to {}",
        output_file_path.to_str().unwrap()
    );
}

pub fn get_function_name_mapper() -> FunctionNameMapper {
    let function_name_mapper_file_path = BASE_DATASET_PATH.join("function_name_mapper.jsonl");
    if !function_name_mapper_file_path.exists() {
        generate_function_name_mapper_file();
    }
    let function_name_mapper_entries = load_json_lines(&function_name_mapper_file_path)
        .expect("Failed to open function name mapper file");
    let function_name_mapper_entries_parsed: Vec<FunctionNameMapperEntry> =
        function_name_mapper_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry).expect("Function name mapper entry has wrong format")
            })
            .collect();
    let original_to_sanitized: HashMap<String, String> = function_name_mapper_entries_parsed
        .into_iter()
        .map(|entry| (entry.original_function_name, entry.sanitized_function_name))
        .collect();
    let sanitized_to_original: HashMap<String, String> = original_to_sanitized
        .iter()
        .map(|(original, sanitized)| (sanitized.clone(), original.clone()))
        .collect();
    FunctionNameMapper {
        original_to_sanitized,
        sanitized_to_original,
    }
}

// fn generate_model_specific_tool_definitions(model: Model, bfcl_tool_definitions: &Vec<BfclFunctionDef>, function_name_mapper: &FunctionNameMapper) -> serde_json::Value {
//     let model_interface = get_model_interface(model);
//     model_interface.generate_tool_definitions(bfcl_tool_definitions, function_name_mapper)
// }
