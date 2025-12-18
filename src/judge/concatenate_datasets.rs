use std::{collections::HashSet, path::Path};

use indexmap::IndexMap;
use pyo3::pyfunction;

use crate::{
    judge::{
        generate_dataset::{
            OneAnswerEntry, PerplexityDatasetMaskEntry, TwoAnswersEntry,
            generate_one_answer_dataset, generate_perplexity_dataset_mask,
            generate_two_answers_dataset,
        },
        result_file_model::{PerplexityResultEntry, PreferenceResultEntry},
    },
    utils::{load_json_lines, write_json_lines_to_file},
};

// #[pyfunction]
// pub fn concatenate_experiment_datasets(config: JudgeConfig, output_file_path: &str) {
//     println!("Concatenating datasets for config: {:?}", config);
//     let model_name = config.model.to_string();
//     let model_safe_name = get_model_directory_safe_name(&model_name);
//     match &config.experiment {
//         JudgeExperiment::Perplexity { lang } => {
//             concatenate_perplexity_datasets(&model_safe_name, lang, output_file_path);
//         }
//         JudgeExperiment::PreferenceDirect { lang1, lang2 } => {
//             concatenate_preference_datasets(&model_safe_name, lang1, lang2, output_file_path);
//         }
//     }
// }

#[pyfunction]
pub fn concatenate_perplexity_datasets(model_safe_name: &str, lang: &str, output_file_path: &str) {
    let correct_dataset_path = format!("judge/datasets/one_answer/{}_correct.jsonl", lang);
    let incorrect_dataset_path = format!("judge/datasets/one_answer/{}_incorrect.jsonl", lang);
    let correct_result_path = format!(
        "judge/result/{}/perplexity/{}_correct.jsonl",
        model_safe_name, lang
    );
    let incorrect_result_path = format!(
        "judge/result/{}/perplexity/{}_incorrect.jsonl",
        model_safe_name, lang
    );
    let mask_path = "judge/datasets/perplexity_mask.jsonl";
    let mut combined_entries: Vec<serde_json::Value> = Vec::new();
    if !Path::new(&correct_dataset_path).exists() || !Path::new(&incorrect_dataset_path).exists() {
        println!(
            "One answer datasets for language {} not found. Generating...",
            lang
        );
        generate_one_answer_dataset(&lang);
    }
    if !Path::new(&mask_path).exists() {
        println!("Perplexity mask dataset not found. Generating...");
        generate_perplexity_dataset_mask();
    }
    let correct_dataset_entries =
        load_json_lines(&correct_dataset_path).expect("Failed to load correct one answer dataset");
    let incorrect_dataset_entries = load_json_lines(&incorrect_dataset_path)
        .expect("Failed to load incorrect one answer dataset");

    let perplexity_mask_entries =
        load_json_lines(&mask_path).expect("Failed to load perplexity mask dataset");

    // parse all entries
    let correct_dataset_entries_parsed: IndexMap<usize, OneAnswerEntry> = correct_dataset_entries
        .into_iter()
        .map(|entry| {
            let parsed: OneAnswerEntry =
                serde_json::from_value(entry).expect("Failed to parse correct one answer entry");
            (parsed.index, parsed)
        })
        .collect();
    let incorrect_dataset_entries_parsed: IndexMap<usize, OneAnswerEntry> =
        incorrect_dataset_entries
            .into_iter()
            .map(|entry| {
                let parsed: OneAnswerEntry = serde_json::from_value(entry)
                    .expect("Failed to parse incorrect one answer entry");
                (parsed.index, parsed)
            })
            .collect();
    // let correct_result_entries =
    //     load_json_lines(&correct_result_path).expect("Failed to load correct one answer result file");
    // let incorrect_result_entries = load_json_lines(&incorrect_result_path)
    //     .expect("Failed to load incorrect one answer result file");
    let correct_result_ids: HashSet<usize> = match load_json_lines(&correct_result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed: PerplexityResultEntry = serde_json::from_value(entry)
                    .expect("Failed to parse correct one answer result entry");
                parsed.index
            })
            .collect(),
        Err(_) => {
            println!(
                "File {} does not exist, assuming no completed entries.",
                correct_result_path
            );
            HashSet::new()
        }
    };
    let incorrect_result_ids: HashSet<usize> = match load_json_lines(&incorrect_result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed: PerplexityResultEntry = serde_json::from_value(entry)
                    .expect("Failed to parse incorrect one answer result entry");
                parsed.index
            })
            .collect(),
        Err(_) => {
            println!(
                "File {} does not exist, assuming no completed entries.",
                incorrect_result_path
            );
            HashSet::new()
        }
    };
    let perplexity_mask_entries_parsed: IndexMap<usize, PerplexityDatasetMaskEntry> =
        perplexity_mask_entries
            .into_iter()
            .map(|entry| {
                let parsed: PerplexityDatasetMaskEntry =
                    serde_json::from_value(entry).expect("Failed to parse perplexity mask entry");
                (parsed.index, parsed)
            })
            .collect();
    let dataset_length = correct_dataset_entries_parsed.len();
    assert_eq!(dataset_length, incorrect_dataset_entries_parsed.len());
    assert_eq!(dataset_length, perplexity_mask_entries_parsed.len());
    let indices = correct_dataset_entries_parsed.keys();
    for index in indices {
        let mask_entry = &perplexity_mask_entries_parsed[*index];
        // only push valid entries
        if mask_entry.valid {
            if !correct_result_ids.contains(index) {
                let correct_entry = correct_dataset_entries_parsed
                    .get(index)
                    .expect("Missing correct one answer entry");
                combined_entries.push(
                    serde_json::to_value(correct_entry)
                        .expect("Failed to serialize correct one answer entry"),
                );
            }
            if !incorrect_result_ids.contains(index) {
                let incorrect_entry = incorrect_dataset_entries_parsed
                    .get(index)
                    .expect("Missing incorrect one answer entry");
                combined_entries.push(
                    serde_json::to_value(incorrect_entry)
                        .expect("Failed to serialize incorrect one answer entry"),
                );
            }
        }
    }
    // serialize combined entries and write to output file
    let combined_entries_serialized: Vec<serde_json::Value> =
        combined_entries.into_iter().map(|entry| entry).collect();
    write_json_lines_to_file(output_file_path, &combined_entries_serialized)
        .expect("Failed to write combined perplexity dataset");
    println!(
        "Concatenated perplexity dataset for language {} written to {}",
        lang, output_file_path
    );
}

#[pyfunction]
pub fn concatenate_preference_datasets(
    model_safe_name: &str,
    lang1: &str,
    lang2: &str,
    output_file_path: &str,
) {
    let lang1_correct_lang2_incorrect_dataset_path = format!(
        "judge/datasets/two_answers/{}_correct_{}_incorrect.jsonl",
        lang1, lang2
    );
    let lang1_incorrect_lang2_correct_dataset_path = format!(
        "judge/datasets/two_answers/{}_incorrect_{}_correct.jsonl",
        lang1, lang2
    );
    let both_correct_dataset_path = format!(
        "judge/datasets/two_answers/{}_correct_{}_correct.jsonl",
        lang1, lang2
    );
    let both_incorrect_dataset_path = format!(
        "judge/datasets/two_answers/{}_incorrect_{}_incorrect.jsonl",
        lang1, lang2
    );
    let lang1_correct_lang2_incorrect_result_path = format!(
        "judge/result/{}/preference/{}_correct_{}_incorrect.jsonl",
        model_safe_name, lang1, lang2
    );
    let lang1_incorrect_lang2_correct_result_path = format!(
        "judge/result/{}/preference/{}_incorrect_{}_correct.jsonl",
        model_safe_name, lang1, lang2
    );
    let both_correct_result_path = format!(
        "judge/result/{}/preference/{}_correct_{}_correct.jsonl",
        model_safe_name, lang1, lang2
    );
    let both_incorrect_result_path = format!(
        "judge/result/{}/preference/{}_incorrect_{}_incorrect.jsonl",
        model_safe_name, lang1, lang2
    );
    let output_paths_exist = [
        &lang1_correct_lang2_incorrect_dataset_path,
        &lang1_incorrect_lang2_correct_dataset_path,
        &both_correct_dataset_path,
        &both_incorrect_dataset_path,
    ]
    .iter()
    .all(|path| Path::new(path).exists());
    if !output_paths_exist {
        println!(
            "Two answers datasets for languages {} and {} not found. Generating...",
            lang1, lang2
        );
        generate_two_answers_dataset(&lang1, &lang2);
    }
    let lang1_correct_lang2_incorrect_dataset_entries =
        load_json_lines(&lang1_correct_lang2_incorrect_dataset_path)
            .expect("Failed to load lang1 correct lang2 incorrect dataset");
    let lang1_incorrect_lang2_correct_dataset_entries =
        load_json_lines(&lang1_incorrect_lang2_correct_dataset_path)
            .expect("Failed to load lang1 incorrect lang2 correct dataset");
    let both_correct_dataset_entries =
        load_json_lines(&both_correct_dataset_path).expect("Failed to load both correct dataset");
    let both_incorrect_dataset_entries = load_json_lines(&both_incorrect_dataset_path)
        .expect("Failed to load both incorrect dataset");

    let lang1_correct_lang2_incorrect_dataset_parsed: Vec<TwoAnswersEntry> =
        lang1_correct_lang2_incorrect_dataset_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry)
                    .expect("Failed to parse lang1 correct lang2 incorrect dataset entry")
            })
            .collect();
    let lang1_incorrect_lang2_correct_dataset_parsed: Vec<TwoAnswersEntry> =
        lang1_incorrect_lang2_correct_dataset_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry)
                    .expect("Failed to parse lang1 incorrect lang2 correct dataset entry")
            })
            .collect();
    let both_correct_dataset_parsed: Vec<TwoAnswersEntry> = both_correct_dataset_entries
        .into_iter()
        .map(|entry| {
            serde_json::from_value(entry).expect("Failed to parse both correct dataset entry")
        })
        .collect();
    let both_incorrect_dataset_parsed: Vec<TwoAnswersEntry> = both_incorrect_dataset_entries
        .into_iter()
        .map(|entry| {
            serde_json::from_value(entry).expect("Failed to parse both incorrect dataset entry")
        })
        .collect();

    let lang1_correct_lang2_incorrect_result_ids =
        match load_json_lines(&lang1_correct_lang2_incorrect_result_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed: PreferenceResultEntry = serde_json::from_value(entry)
                        .expect("Failed to parse lang1 correct lang2 incorrect result entry");
                    parsed.index
                })
                .collect(),
            Err(_) => {
                println!(
                    "File {} does not exist, assuming no completed entries.",
                    lang1_correct_lang2_incorrect_result_path
                );
                HashSet::new()
            }
        };
    let lang1_incorrect_lang2_correct_result_ids =
        match load_json_lines(&lang1_incorrect_lang2_correct_result_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed: PreferenceResultEntry = serde_json::from_value(entry)
                        .expect("Failed to parse lang1 incorrect lang2 correct result entry");
                    parsed.index
                })
                .collect(),
            Err(_) => {
                println!(
                    "File {} does not exist, assuming no completed entries.",
                    lang1_incorrect_lang2_correct_result_path
                );
                HashSet::new()
            }
        };
    let both_correct_result_ids = match load_json_lines(&both_correct_result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed: PreferenceResultEntry = serde_json::from_value(entry)
                    .expect("Failed to parse both correct result entry");
                parsed.index
            })
            .collect(),
        Err(_) => {
            println!(
                "File {} does not exist, assuming no completed entries.",
                both_correct_result_path
            );
            HashSet::new()
        }
    };
    let both_incorrect_result_ids = match load_json_lines(&both_incorrect_result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed: PreferenceResultEntry = serde_json::from_value(entry)
                    .expect("Failed to parse both incorrect result entry");
                parsed.index
            })
            .collect(),
        Err(_) => {
            println!(
                "File {} does not exist, assuming no completed entries.",
                both_incorrect_result_path
            );
            HashSet::new()
        }
    };

    // conditionally concatenate
    let mut combined_entries: Vec<serde_json::Value> = Vec::new();
    // combined_entries.extend(lang1_correct_lang2_incorrect_dataset_entries);
    // combined_entries.extend(lang1_incorrect_lang2_correct_dataset_entries);
    // combined_entries.extend(both_correct_dataset_entries);
    // combined_entries.extend(both_incorrect_dataset_entries);
    for entry in lang1_correct_lang2_incorrect_dataset_parsed {
        if !lang1_correct_lang2_incorrect_result_ids.contains(&entry.index) {
            combined_entries.push(
                serde_json::to_value(&entry)
                    .expect("Failed to serialize lang1 correct lang2 incorrect dataset entry"),
            );
        }
    }
    for entry in lang1_incorrect_lang2_correct_dataset_parsed {
        if !lang1_incorrect_lang2_correct_result_ids.contains(&entry.index) {
            combined_entries.push(
                serde_json::to_value(&entry)
                    .expect("Failed to serialize lang1 incorrect lang2 correct dataset entry"),
            );
        }
    }
    for entry in both_correct_dataset_parsed {
        if !both_correct_result_ids.contains(&entry.index) {
            combined_entries.push(
                serde_json::to_value(&entry)
                    .expect("Failed to serialize both correct dataset entry"),
            );
        }
    }
    for entry in both_incorrect_dataset_parsed {
        if !both_incorrect_result_ids.contains(&entry.index) {
            combined_entries.push(
                serde_json::to_value(&entry)
                    .expect("Failed to serialize both incorrect dataset entry"),
            );
        }
    }
    // write to output file
    write_json_lines_to_file(output_file_path, &combined_entries)
        .expect("Failed to write combined preference direct dataset");
    println!(
        "Concatenated preference direct dataset for languages {} and {} written to {}",
        lang1, lang2, output_file_path
    );
}
