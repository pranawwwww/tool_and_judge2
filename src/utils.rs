use crate::{
    tool::bfcl_formats::{BfclDatasetEntry, BfclGroundTruthEntry},
    tool::file_models::{CategorizedEntry, EvaluationResultEntry, InferenceJsonEntry, InferenceRawEntry},
};
use serde_json::Value;
    use std::{fs::File, io::{BufRead, BufReader}};
pub fn load_json_lines2(file: File) -> Result<Vec<serde_json::Value>, String> {
    

    let reader = BufReader::new(file);

    let mut results = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Unable to read line: {}", e))?;
        let line_json: Value =
            serde_json::from_str(&line).map_err(|e| format!("Unable to parse JSON: {}", e))?;
        results.push(line_json);
    }
    Ok(results)
}

pub fn load_json_lines(file_path: &str) -> Result<Vec<serde_json::Value>, String> {

    let file = File::open(file_path).map_err(|e| format!("Unable to open file: {}", e))?;
    // let reader = BufReader::new(file);

    // let mut results = Vec::new();

    // for line in reader.lines() {
    //     let line = line.map_err(|e| format!("Unable to read line: {}", e))?;
    //     let line_json: Value =
    //         serde_json::from_str(&line).map_err(|e| format!("Unable to parse JSON: {}", e))?;
    //     results.push(line_json);
    // }
    let results = load_json_lines2(file)?;
    Ok(results)
}


pub fn load_test_cases(file_path: &str) -> Result<Vec<BfclDatasetEntry>, String> {
    let json_lines = load_json_lines(file_path)?;
    let test_cases = deserialize_test_cases(json_lines);
    Ok(test_cases)
}

// Try to load test cases and their IDs, returning both.
pub fn try_load_test_cases_and_ids(file_path: &str) -> (Vec<BfclDatasetEntry>, Vec<String>) {
    let test_cases = match load_test_cases(file_path) {
        Ok(cases) => cases,
        Err(_) => {
            println!("File {} does not exist, it will be created.", file_path);
            Vec::new()
        }
    };
    let existing_ids = test_cases
        .iter()
        .map(|entry| entry.id.clone())
        .collect::<Vec<String>>();
    (test_cases, existing_ids)
}

pub fn try_load_inference_raw_and_ids(file_path: &str) -> (Vec<InferenceRawEntry>, Vec<String>) {
    let inference_entries = match load_json_lines(file_path) {
        Ok(json_lines) => deserialize_inference_raw_entries(json_lines),
        Err(_) => {
            println!("File {} does not exist, it will be created.", file_path);
            Vec::new()
        }
    };
    let existing_ids = inference_entries
        .iter()
        .map(|entry| entry.id.clone())
        .collect::<Vec<String>>();
    (inference_entries, existing_ids)
}

pub fn try_load_inference_json_and_ids(file_path: &str) -> (Vec<InferenceJsonEntry>, Vec<String>) {
    let inference_entries = match load_json_lines(file_path) {
        Ok(json_lines) => deserialize_inference_json_entries(json_lines),
        Err(_) => {
            println!("File {} does not exist, it will be created.", file_path);
            Vec::new()
        }
    };
    let existing_ids = inference_entries
        .iter()
        .map(|entry| entry.id.clone())
        .collect::<Vec<String>>();
    (inference_entries, existing_ids)
}

pub fn write_json_lines_to_file(
    file_path: &str,
    results: &Vec<serde_json::Value>,
) -> Result<(), String> {
    use std::fs::{File, create_dir_all};
    use std::io::Write;
    use std::path::Path;

    let path = Path::new(file_path);
    if let Some(parent) = path.parent() {
        create_dir_all(parent).map_err(|e| format!("Unable to create parent directory: {}", e))?;
    }

    let mut file = File::create(file_path).map_err(|e| format!("Unable to create file: {}", e))?;

    for result in results {
        let line = serde_json::to_string(result)
            .map_err(|e| format!("Unable to serialize JSON: {}", e))?;
        writeln!(file, "{}", line).map_err(|e| format!("Unable to write to file: {}", e))?;
    }
    file.flush()
        .map_err(|e| format!("Unable to flush file: {}", e))?;

    Ok(())
}

pub fn serialize_test_cases(test_cases: &Vec<BfclDatasetEntry>) -> Vec<serde_json::Value> {
    test_cases
        .iter()
        .map(|case| {
            serde_json::to_value(case).expect("Unable to serialize BfclDatasetEntry")
        })
        .collect()
}

pub fn serialize_inference_raw_entries(
    inference_raw_entries: &Vec<InferenceRawEntry>,
) -> Vec<serde_json::Value> {
    inference_raw_entries
        .iter()
        .map(|entry| serde_json::to_value(entry).expect("Unable to serialize InferenceRawEntry"))
        .collect()
}
pub fn serialize_inference_json_entries(
    inference_json_entries: &Vec<InferenceJsonEntry>,
) -> Vec<serde_json::Value> {
    inference_json_entries
        .iter()
        .map(|entry| serde_json::to_value(entry).expect("Unable to serialize InferenceJsonEntry"))
        .collect()
}

pub fn serialize_evaluation_result_entries(
    evaluation_result_entries: &Vec<EvaluationResultEntry>,
) -> Vec<serde_json::Value> {
    evaluation_result_entries
        .iter()
        .map(|entry| {
            serde_json::to_value(entry).expect("Unable to serialize EvaluationResultEntry")
        })
        .collect()
}

pub fn serialize_categorized_entries(
    categorized_entries: &Vec<CategorizedEntry>,
) -> Vec<serde_json::Value> {
    categorized_entries
        .iter()
        .map(|entry| {
            serde_json::to_value(entry).expect("Unable to serialize CategorizedEntry")
        })
        .collect()
}

pub fn compare_id(id1: &str, id2: &str) -> std::cmp::Ordering {
    use regex::Regex;
    let re = Regex::new(r"\d+").unwrap();
    let num1 = re
        .find(id1)
        .and_then(|mat| mat.as_str().parse::<u32>().ok())
        .unwrap_or(u32::MAX);
    let num2 = re
        .find(id2)
        .and_then(|mat| mat.as_str().parse::<u32>().ok())
        .unwrap_or(u32::MAX);
    num1.cmp(&num2)
}

pub fn get_model_directory_safe_name(model_name: &str) -> String {
    model_name.replace("/", "-").replace(":", "-")
}

pub fn deserialize_test_cases(cases_to_translate: Vec<serde_json::Value>) -> Vec<BfclDatasetEntry> {
    cases_to_translate
        .iter()
        .map(|case| {
            serde_json::from_value(case.clone()).expect("Test case has wrong format")
        })
        .collect()
}

pub fn deserialize_inference_json_entries(
    inference_json_entries: Vec<serde_json::Value>,
) -> Vec<InferenceJsonEntry> {
    inference_json_entries
        .iter()
        .map(|entry| {
            serde_json::from_value(entry.clone()).expect("Inference JSON entry has wrong format")
        })
        .collect()
}

pub fn deserialize_inference_raw_entries(
    inference_raw_entries: Vec<serde_json::Value>,
) -> Vec<InferenceRawEntry> {
    inference_raw_entries
        .iter()
        .map(|entry| {
            serde_json::from_value(entry.clone()).expect("Inference raw entry has wrong format")
        })
        .collect()
}

pub fn deserialize_ground_truth_entries(
    ground_truth_entries: Vec<serde_json::Value>,
) -> Vec<BfclGroundTruthEntry> {
    ground_truth_entries
        .iter()
        .map(|entry| {
            serde_json::from_value(entry.clone())
                .expect("Ground truth entry has wrong format")
        })
        .collect()
}

pub fn deserialize_evaluation_result_entries(
    evaluation_result_entries: Vec<serde_json::Value>,
) -> Vec<EvaluationResultEntry> {
    evaluation_result_entries
        .iter()
        .map(|entry| {
            serde_json::from_value(entry.clone())
                .expect("Evaluation result entry has wrong format")
        })
        .collect()
}
pub fn deserialize_categorized_entries(
    categorized_entries: Vec<serde_json::Value>,
) -> Vec<CategorizedEntry> {
    categorized_entries
        .iter()
        .map(|entry| {
            serde_json::from_value(entry.clone()).expect("Categorized entry has wrong format")
        })
        .collect()
}