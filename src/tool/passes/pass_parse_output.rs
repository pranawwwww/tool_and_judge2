use pyo3::pyfunction;
use serde::{Deserialize, Serialize};

use crate::{
    config::ToolConfig,
    models::model_interface::get_model_interface,
    tool::{
        base_path::BASE_RESULT_PATH, bfcl_formats::BfclOutputFunctionCall,
        error_analysis::EvaluationError, experiments::GenerateRawFileName,
        passes::pass_generate_raw::{GenerateRawEntry, get_function_name_mapper},
    },
    utils::{get_model_safe_name, load_json_lines, write_json_lines_to_file},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct ParseOutputEntry {
    pub id: String,
    pub valid: bool,
    pub result: Result<Vec<BfclOutputFunctionCall>, EvaluationError>,
}

#[pyfunction]
pub fn pass_parse_output(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let model_interface = get_model_interface(model);
    let function_name_mapper = get_function_name_mapper();
    for experiment in config.experiments.iter() {
        let generate_raw_file_name = GenerateRawFileName::from_config_experiment(experiment);
        let generate_raw_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&generate_raw_file_name).unwrap()
        );
        let generate_raw_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("generate_raw")
            .join(&generate_raw_file_name_str);
        let parse_output_file_name_str = generate_raw_file_name_str.clone();
        let parse_output_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("parse_output")
            .join(&parse_output_file_name_str);
        let generate_raw_entries = load_json_lines(&generate_raw_file_path)
            .expect("Failed to load generate raw output file");
        let generate_raw_entries_parsed: Vec<GenerateRawEntry> = generate_raw_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry).expect("Generate raw output entry has wrong format")
            })
            .collect();
        let mut parse_output_entries: Vec<ParseOutputEntry> = Vec::new();
        for raw_entry in generate_raw_entries_parsed.iter() {
            let parsed_result = model_interface.parse_tool_calls(
                &raw_entry.raw_output,
                &function_name_mapper,
            );
            let parse_output_entry = ParseOutputEntry {
                id: raw_entry.id.clone(),
                valid: parsed_result.is_ok(),
                result: parsed_result,
            };
            parse_output_entries.push(parse_output_entry);
        }
        let parsed_output_entries_serialized: Vec<serde_json::Value> = parse_output_entries
            .into_iter()
            .map(|entry| serde_json::to_value(entry).unwrap())
            .collect();
        write_json_lines_to_file(parse_output_file_path, &parsed_output_entries_serialized)
            .expect("Failed to write parse output entries to file");
    }
    println!("Pass parse output completed for model {}", model_safe_name);
}
