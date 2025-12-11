


pub fn load_json_lines_with_id(file_path: &str) -> Result<(Vec<serde_json::Value>, Vec<String>), String>{
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use serde_json::Value;

    let file = File::open(file_path).map_err(|e| format!("Unable to open file: {}", e))?;
    let reader = BufReader::new(file);

    let mut results = Vec::new();
    let mut existing_ids = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Unable to read line: {}", e))?;
        let line_json: Value = serde_json::from_str(&line).map_err(|e| format!("Unable to parse JSON: {}", e))?;
        if let Some(id) = line_json.get("id") {
            if let Some(id_str) = id.as_str() {
                existing_ids.push(id_str.to_string());
            }
        }
        results.push(line_json);
    }

    Ok((results, existing_ids))
}

pub fn load_json_lines(file_path: &str) -> Result<Vec<serde_json::Value>, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use serde_json::Value;

    let file = File::open(file_path).map_err(|e| format!("Unable to open file: {}", e))?;
    let reader = BufReader::new(file);

    let mut results = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Unable to read line: {}", e))?;
        let line_json: Value = serde_json::from_str(&line).map_err(|e| format!("Unable to parse JSON: {}", e))?;
        results.push(line_json);
    }

    Ok(results)
}

pub fn write_json_lines_to_file(file_path: &str, results: &Vec<serde_json::Value>) -> Result<(), String> {
    use std::fs::{File, create_dir_all};
    use std::io::Write;
    use std::path::Path;

    let path = Path::new(file_path);
    if let Some(parent) = path.parent() {
        create_dir_all(parent).map_err(|e| format!("Unable to create parent directory: {}", e))?;
    }

    let mut file = File::create(file_path).map_err(|e| format!("Unable to create file: {}", e))?;

    for result in results {
        let line = serde_json::to_string(result).map_err(|e| format!("Unable to serialize JSON: {}", e))?;
        writeln!(file, "{}", line).map_err(|e| format!("Unable to write to file: {}", e))?;
    }
    file.flush().map_err(|e| format!("Unable to flush file: {}", e))?;

    Ok(())
}


pub fn sort_results_by_id(results: &mut Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    use regex::Regex;
    let re = Regex::new(r"\d+").unwrap();
    results.sort_by_key(|x| {
        if let Some(id_value) = x.get("id") {
            if let Some(id_str) = id_value.as_str() {
                if let Some(mat) = re.find(id_str) {
                    return mat.as_str().parse::<u32>().unwrap_or(u32::MAX);
                }
            }
        }
        u32::MAX
    });
    results.to_vec()
}

pub fn sort_and_write_json_lines(file_path: &str, results: &mut Vec<serde_json::Value>) -> Result<(), String> {
    let sorted_results = sort_results_by_id(results);
    write_json_lines_to_file(file_path, &sorted_results)
}

pub fn get_model_directory_safe_name(model_name: &str) -> String {
    model_name.replace("/", "-").replace(":", "-")
}

