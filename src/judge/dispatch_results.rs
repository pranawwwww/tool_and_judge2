use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use pyo3::pyfunction;

use crate::{
    judge::{
        generate_dataset::{get_preference_indices, get_valid_perplexity_indices},
        result_file_model::PerplexityResultEntry,
    },
    utils::{load_json_lines, write_json_lines_to_file},
};

#[pyfunction]
pub fn dispatch_perplexity_results(model_safe_name: &str, lang: &str, input_file_path: &str) {
    println!(
        "Dispatching perplexity results for model: {}, lang: {}, input file: {}",
        model_safe_name, lang, input_file_path
    );
    let correct_result_path = format!(
        "judge/result/{}/perplexity/{}_correct.jsonl",
        model_safe_name, lang
    );
    let incorrect_result_path = format!(
        "judge/result/{}/perplexity/{}_incorrect.jsonl",
        model_safe_name, lang
    );
    let mut correct_result_entries: Vec<PerplexityResultEntry> =
        match load_json_lines(&correct_result_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<PerplexityResultEntry>(entry)
                        .expect("Failed to parse correct perplexity entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    let mut incorrect_result_entries: Vec<PerplexityResultEntry> =
        match load_json_lines(&incorrect_result_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<PerplexityResultEntry>(entry)
                        .expect("Failed to parse incorrect perplexity entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    if !Path::new(input_file_path).exists() {
        println!(
            "Input file does not exist: {}, skipping dispatching.",
            input_file_path
        );
        return;
    }
    let combined_entries = load_json_lines(input_file_path).expect("Failed to load input file");
    let combined_entries_parsed: HashMap<(usize, bool), PerplexityResultEntry> = combined_entries
        .into_iter()
        .map(|entry| {
            let parsed = serde_json::from_value::<PerplexityResultEntry>(entry)
                .expect("Failed to parse combined perplexity entry");
            ((parsed.index, parsed.is_correct), parsed)
        })
        .collect();
    let mut remaining_correct_indices: HashSet<usize> = get_valid_perplexity_indices();
    let mut remainint_incorrect_indices: HashSet<usize> = remaining_correct_indices.clone();
    for entry in correct_result_entries.iter() {
        let result = remaining_correct_indices.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in correct results: {}",
            entry.index
        );
    }
    for entry in incorrect_result_entries.iter() {
        let result = remainint_incorrect_indices.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in incorrect results: {}",
            entry.index
        );
    }
    let mut missing_correct_index_count = 0;
    let mut missing_incorrect_index_count = 0;
    for index in remaining_correct_indices {
        let key = (index, true);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            correct_result_entries.push(entry.clone());
        } else {
            missing_correct_index_count += 1;
        }
    }
    for index in remainint_incorrect_indices {
        let key = (index, false);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            incorrect_result_entries.push(entry.clone());
        } else {
            missing_incorrect_index_count += 1;
        }
    }
    if missing_correct_index_count > 0 {
        println!(
            "Warning: Missing {} correct entries for model: {}, lang: {}",
            missing_correct_index_count, model_safe_name, lang
        );
    }
    if missing_incorrect_index_count > 0 {
        println!(
            "Warning: Missing {} incorrect entries for model: {}, lang: {}",
            missing_incorrect_index_count, model_safe_name, lang
        );
    }
    correct_result_entries.sort_by_key(|e| e.index);
    incorrect_result_entries.sort_by_key(|e| e.index);
    let correct_serialized = correct_result_entries
        .iter()
        .map(|e| serde_json::to_value(e).expect("Failed to serialize correct perplexity entry"))
        .collect::<Vec<_>>();
    let incorrect_serialized = incorrect_result_entries
        .iter()
        .map(|e| serde_json::to_value(e).expect("Failed to serialize incorrect perplexity entry"))
        .collect::<Vec<_>>();
    write_json_lines_to_file(&correct_result_path, &correct_serialized)
        .expect("Failed to write json lines");
    write_json_lines_to_file(&incorrect_result_path, &incorrect_serialized)
        .expect("Failed to write json lines");
    println!(
        "Dispatched perplexity results for model: {}, lang: {}",
        model_safe_name, lang
    );
}

pub fn dispatch_preference_results(
    model_safe_name: &str,
    lang1: &str,
    lang2: &str,
    input_file_path: &str,
) {
    println!(
        "Dispatching preference results for model: {}, lang1: {}, lang2: {}, input file: {}",
        model_safe_name, lang1, lang2, input_file_path
    );
    let lang1_correct_lang2_incorrect_path = format!(
        "judge/result/{}/preference/{}_correct_{}_incorrect.jsonl",
        model_safe_name, lang1, lang2
    );
    let lang1_incorrect_lang2_correct_path = format!(
        "judge/result/{}/preference/{}_incorrect_{}_correct.jsonl",
        model_safe_name, lang1, lang2
    );
    let both_correct_path = format!(
        "judge/result/{}/preference/{}_correct_{}_correct.jsonl",
        model_safe_name, lang1, lang2
    );
    let both_incorrect_path = format!(
        "judge/result/{}/preference/{}_incorrect_{}_incorrect.jsonl",
        model_safe_name, lang1, lang2
    );
    let mut lang1_correct_lang2_incorrect_entries: Vec<
        crate::judge::result_file_model::PreferenceResultEntry,
    > = match load_json_lines(&lang1_correct_lang2_incorrect_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed = serde_json::from_value::<
                    crate::judge::result_file_model::PreferenceResultEntry,
                >(entry)
                .expect("Failed to parse lang1 correct lang2 incorrect preference entry");
                parsed
            })
            .collect(),
        Err(e) => {
            println!("Cannot open file: {}, assuming empty result file", e);
            vec![]
        }
    };
    let mut lang1_incorrect_lang2_correct_entries: Vec<
        crate::judge::result_file_model::PreferenceResultEntry,
    > = match load_json_lines(&lang1_incorrect_lang2_correct_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed = serde_json::from_value::<
                    crate::judge::result_file_model::PreferenceResultEntry,
                >(entry)
                .expect("Failed to parse lang1 incorrect lang2 correct preference entry");
                parsed
            })
            .collect(),
        Err(e) => {
            println!("Cannot open file: {}, assuming empty result file", e);
            vec![]
        }
    };
    let mut both_correct_entries: Vec<crate::judge::result_file_model::PreferenceResultEntry> =
        match load_json_lines(&both_correct_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<
                        crate::judge::result_file_model::PreferenceResultEntry,
                    >(entry)
                    .expect("Failed to parse both correct preference entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    let mut both_incorrect_entries: Vec<crate::judge::result_file_model::PreferenceResultEntry> =
        match load_json_lines(&both_incorrect_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<
                        crate::judge::result_file_model::PreferenceResultEntry,
                    >(entry)
                    .expect("Failed to parse both incorrect preference entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    if !Path::new(input_file_path).exists() {
        println!(
            "Input file does not exist: {}, skipping dispatching.",
            input_file_path
        );
        return;
    }
    let combined_entries = load_json_lines(input_file_path).expect("Failed to load input file");
    let combined_entries_parsed: HashMap<
        (usize, bool, bool),
        crate::judge::result_file_model::PreferenceResultEntry,
    > = combined_entries
        .into_iter()
        .map(|entry| {
            let parsed = serde_json::from_value::<
                crate::judge::result_file_model::PreferenceResultEntry,
            >(entry)
            .expect("Failed to parse combined preference entry");
            let is_correct1 = parsed.is_correct1;
            let is_correct2 = parsed.is_correct2;
            ((parsed.index, is_correct1, is_correct2), parsed)
        })
        .collect();
    let mut remaining_indices_lang1_correct_lang2_incorrect: HashSet<usize> =
        get_preference_indices();
    let mut remaining_indices_lang1_incorrect_lang2_correct: HashSet<usize> =
        remaining_indices_lang1_correct_lang2_incorrect.clone();
    let mut remaining_indices_both_correct: HashSet<usize> =
        remaining_indices_lang1_incorrect_lang2_correct.clone();
    let mut remaining_indices_both_incorrect: HashSet<usize> =
        remaining_indices_both_correct.clone();
    for entry in lang1_correct_lang2_incorrect_entries.iter() {
        let result = remaining_indices_lang1_correct_lang2_incorrect.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in lang1 correct lang2 incorrect results: {}",
            entry.index
        );
    }
    for entry in lang1_incorrect_lang2_correct_entries.iter() {
        let result = remaining_indices_lang1_incorrect_lang2_correct.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in lang1 incorrect lang2 correct results: {}",
            entry.index
        );
    }
    for entry in both_correct_entries.iter() {
        let result = remaining_indices_both_correct.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in both correct results: {}",
            entry.index
        );
    }
    for entry in both_incorrect_entries.iter() {
        let result = remaining_indices_both_incorrect.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in both incorrect results: {}",
            entry.index
        );
    }
    let mut missing_lang1_correct_lang2_incorrect_count = 0;
    let mut missing_lang1_incorrect_lang2_correct_count = 0;
    let mut missing_both_correct_count = 0;
    let mut missing_both_incorrect_count = 0;
    for index in remaining_indices_lang1_correct_lang2_incorrect {
        let key = (index, true, false);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            lang1_correct_lang2_incorrect_entries.push(entry.clone());
        } else {
            missing_lang1_correct_lang2_incorrect_count += 1;
        }
    }
    for index in remaining_indices_lang1_incorrect_lang2_correct {
        let key = (index, false, true);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            lang1_incorrect_lang2_correct_entries.push(entry.clone());
        } else {
            missing_lang1_incorrect_lang2_correct_count += 1;
        }
    }
    for index in remaining_indices_both_correct {
        let key = (index, true, true);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            both_correct_entries.push(entry.clone());
        } else {
            missing_both_correct_count += 1;
        }
    }
    for index in remaining_indices_both_incorrect {
        let key = (index, false, false);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            both_incorrect_entries.push(entry.clone());
        } else {
            missing_both_incorrect_count += 1;
        }
    }
    if [
        missing_lang1_correct_lang2_incorrect_count,
        missing_lang1_incorrect_lang2_correct_count,
        missing_both_correct_count,
        missing_both_incorrect_count,
    ]
    .iter()
    .any(|&count| count > 0)
    {
        println!(
            "Warning: Missing entries for model: {}, lang1: {}, lang2: {}: lang1 correct lang2 incorrect: {}, lang1 incorrect lang2 correct: {}, both correct: {}, both incorrect: {}",
            model_safe_name,
            lang1,
            lang2,
            missing_lang1_correct_lang2_incorrect_count,
            missing_lang1_incorrect_lang2_correct_count,
            missing_both_correct_count,
            missing_both_incorrect_count
        );
    }
    lang1_correct_lang2_incorrect_entries.sort_by_key(|e| e.index);
    lang1_incorrect_lang2_correct_entries.sort_by_key(|e| e.index);
    both_correct_entries.sort_by_key(|e| e.index);
    both_incorrect_entries.sort_by_key(|e| e.index);
    let lang1_correct_lang2_incorrect_serialized: Vec<serde_json::Value> =
        lang1_correct_lang2_incorrect_entries
            .iter()
            .map(|e| {
                serde_json::to_value(e)
                    .expect("Failed to serialize lang1 correct lang2 incorrect preference entry")
            })
            .collect();
    let lang1_incorrect_lang2_correct_serialized: Vec<serde_json::Value> =
        lang1_incorrect_lang2_correct_entries
            .iter()
            .map(|e| {
                serde_json::to_value(e)
                    .expect("Failed to serialize lang1 incorrect lang2 correct preference entry")
            })
            .collect();
    let both_correct_serialized: Vec<serde_json::Value> = both_correct_entries
        .iter()
        .map(|e| {
            serde_json::to_value(e).expect("Failed to serialize both correct preference entry")
        })
        .collect();
    let both_incorrect_serialized: Vec<serde_json::Value> = both_incorrect_entries
        .iter()
        .map(|e| {
            serde_json::to_value(e).expect("Failed to serialize both incorrect preference entry")
        })
        .collect();
    write_json_lines_to_file(
        &lang1_correct_lang2_incorrect_path,
        &lang1_correct_lang2_incorrect_serialized,
    )
    .expect("Failed to write json lines");
    write_json_lines_to_file(
        &lang1_incorrect_lang2_correct_path,
        &lang1_incorrect_lang2_correct_serialized,
    )
    .expect("Failed to write json lines");
    write_json_lines_to_file(&both_correct_path, &both_correct_serialized)
        .expect("Failed to write json lines");
    write_json_lines_to_file(&both_incorrect_path, &both_incorrect_serialized)
        .expect("Failed to write json lines");
    println!(
        "Dispatched preference results for model: {}, lang1: {}, lang2: {}",
        model_safe_name, lang1, lang2
    );
}
