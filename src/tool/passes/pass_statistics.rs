use std::collections::HashMap;

use indexmap::IndexMap;
use pyo3::pyfunction;
use serde::Serialize;
use strum::IntoEnumIterator;

use crate::{config::ToolConfig, tool::{base_path::BASE_RESULT_PATH, error_analysis::ToolErrorCategory, experiments::EvaluateFileName, passes::{pass_categorize::CategorizeEntry, pass_evaluate::EvaluateEntry}}, utils::{get_model_safe_name, load_json_lines}};

#[derive(Serialize, Clone)]
pub struct Statistics {
    pub accuracy: f32,
    pub total_cases: usize,
    pub correct_cases: usize,
    pub incorrect_cases: usize,
    pub category_counts: IndexMap<ToolErrorCategory, usize>,
    pub category_samples: IndexMap<ToolErrorCategory, Vec<String>>,
}

#[pyfunction]
pub fn pass_statistics(config: &ToolConfig) {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    for experiment in config.experiments.iter() {
        let evaluate_file_name = EvaluateFileName::from_config_experiment(experiment);
        let evaluate_file_name_str = format!(
            "{}.jsonl",
            serde_json::to_string(&evaluate_file_name).unwrap()
        );
        let categorize_file_name_str = evaluate_file_name_str.clone();
        let statistics_file_name_str = format!(
            "{}.json", // Use .json extension for statistics file
            serde_json::to_string(&evaluate_file_name).unwrap()
        );
        let evaluate_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("evaluate")
            .join(&evaluate_file_name_str);
        let categorize_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("categorize")
            .join(&categorize_file_name_str);
        let statistics_file_path = BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("statistics")
            .join(&statistics_file_name_str);
        let evaluate_entries = load_json_lines(&evaluate_file_path)
            .expect("Should load evaluate entries");
        let evaluate_entries_parsed: Vec<EvaluateEntry> = evaluate_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry).expect("Should parse evaluate entry")
            })
            .collect();
        let categorize_entries = load_json_lines(&categorize_file_path)
            .expect("Should load categorize entries");
        let categorize_entries_parsed: Vec<CategorizeEntry> = categorize_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry).expect("Should parse categorize entry")
            })
            .collect();
        let num_total_cases = evaluate_entries_parsed.len();
        let num_incorrect_cases = categorize_entries_parsed.len();
        let num_correct_cases = num_total_cases - num_incorrect_cases;
        let accuracy = if num_total_cases == 0 {
            0.0
        } else {
            num_correct_cases as f32 / num_total_cases as f32
        };
        let mut category_samples: HashMap<ToolErrorCategory, Vec<String>> = HashMap::new();
        for categorize_entry in categorize_entries_parsed.iter() {
            category_samples
                .entry(categorize_entry.error_category.clone())
                .or_insert_with(Vec::new)
                .push(categorize_entry.id.clone());
        }
        let category_samples: IndexMap<ToolErrorCategory, Vec<String>> = ToolErrorCategory::iter()
            .filter_map(|category| {
                if let Some(samples) = category_samples.get(&category) {
                    Some((category, samples.clone()))
                } else {
                    None
                }
            })
            .collect();
        let category_counts: IndexMap<ToolErrorCategory, usize> = ToolErrorCategory::iter()
            .map(|category| {
                let count = category_samples
                    .get(&category)
                    .map(|samples| samples.len())
                    .unwrap_or(0);
                (category, count)
            })
            .collect();
        let statistics = Statistics {
            accuracy,
            total_cases: num_total_cases,
            correct_cases: num_correct_cases,
            incorrect_cases: num_incorrect_cases,
            category_counts,
            category_samples,
        };
        let statistics_serialized = serde_json::to_string_pretty(&statistics)
            .expect("Should serialize statistics");
        std::fs::create_dir_all(
            statistics_file_path
                .parent()
                .expect("Should get parent directory"),
        )
        .expect("Should create statistics directory");
        std::fs::write(&statistics_file_path, statistics_serialized)
            .expect("Should write statistics file");
        println!(
            "Wrote statistics to {}",
            statistics_file_path.display()
        );
    }
}