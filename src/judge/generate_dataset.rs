use std::{collections::HashSet, path::Path};

use indexmap::IndexMap;
use pyo3::{Python, types::PyAnyMethods};
use serde::{Deserialize, Serialize};

use crate::{
    utils::{load_json_lines, write_json_lines_to_file},
};

#[derive(Serialize, Deserialize)]
pub struct MmmluDatasetEntryNormalized {
    pub original_index: usize,
    pub question: String,
    pub choices: [String; 4],
    pub answer: usize,
    pub subject: String,
}
#[derive(Deserialize)]
pub struct MmmluDatasetEntryChinese {
    pub original_index: usize,
    #[serde(rename = "Question")]
    pub question: String,
    #[serde(rename = "A")]
    pub choice_a: String,
    #[serde(rename = "B")]
    pub choice_b: String,
    #[serde(rename = "C")]
    pub choice_c: String,
    #[serde(rename = "D")]
    pub choice_d: String,
    #[serde(rename = "Answer")]
    pub answer: String,
    #[serde(rename = "Subject")]
    pub subject: String,
}

#[derive(Deserialize, Serialize)]
pub struct OneAnswerEntry {
    pub index: usize,
    pub question: String,
    pub answer: String,
    pub lang: String,
    pub is_correct: bool,
    pub subject: String,
}

#[derive(Deserialize, Serialize)]
pub struct TwoAnswersEntry {
    pub index: usize,
    pub question: String,
    pub answer1: String,
    pub answer2: String,
    pub lang1: String,
    pub lang2: String,
    pub is_correct1: bool,
    pub is_correct2: bool,
    pub subject: String,
}

#[derive(Deserialize, Serialize)]
pub struct PerplexityDatasetMaskEntry {
    pub index: usize,
    pub valid: bool,
    pub question: String,
    pub choices: [String; 4],
    pub subject: String,
}

fn download_mmmlu_dataset(lang: &str) {
    Python::attach(|py| {
        let download_mmmlu_module = py
            .import("src_py.judge.download_mmmlu_dataset")
            .expect("Failed to import src_py.judge.download_mmmlu_dataset module");
        let download_mmmlu_func = download_mmmlu_module
            .getattr("download_mmmlu_dataset")
            .expect("Failed to get download_mmmlu_dataset function");
        let output_path = format!("judge/datasets/mmmlu/{}.jsonl", lang);
        download_mmmlu_func
            .call1((lang, output_path))
            .expect("Failed to call download_mmmlu_dataset function");
    });
    println!("Downloaded MMMLU dataset for language {}", lang);
}

// #[pyfunction]
pub fn generate_normalized_datasets(lang: &str) {
    let input_dataset_path = format!("judge/datasets/mmmlu/{}.jsonl", lang);

    if !Path::new(&input_dataset_path).exists() {
        println!(
            "MMMLU dataset for language {} not found. Downloading...",
            lang
        );
        download_mmmlu_dataset(lang);
    }
    let entries = load_json_lines(&input_dataset_path).expect("Failed to load MMMLU dataset");
    let parsed_and_normalized_entries: Vec<MmmluDatasetEntryNormalized> = entries
        .into_iter()
        .map(|entry| parse_and_normalize(&entry, lang))
        .collect();
    let serialized_results: Vec<serde_json::Value> = parsed_and_normalized_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).expect("Failed to serialize normalized entry"))
        .collect();

    // write to output file
    let output_dataset_path = format!("judge/datasets/mmmlu_normalized/{}.jsonl", lang);
    write_json_lines_to_file(&output_dataset_path, &serialized_results)
        .expect("Failed to write normalized dataset");
}

pub fn generate_one_answer_dataset(lang: &str) {
    let output_correct_path = format!("judge/datasets/one_answer/{}_correct.jsonl", lang);
    let output_incorrect_path = format!("judge/datasets/one_answer/{}_incorrect.jsonl", lang);
    let correct_exists = Path::new(&output_correct_path).exists();
    let incorrect_exists = Path::new(&output_incorrect_path).exists();
    if correct_exists && incorrect_exists {
        println!(
            "One answer datasets for language {} already exist. Skipping generation.",
            lang
        );
        return;
    }
    let input_dataset_path = format!("judge/datasets/mmmlu_normalized/{}.jsonl", lang);
    if !Path::new(&input_dataset_path).exists() {
        println!(
            "Normalized MMMLU dataset for language {} not found. Generating...",
            lang
        );
        generate_normalized_datasets(lang);
    }
    let entries =
        load_json_lines(&input_dataset_path).expect("Failed to load normalized MMMLU dataset");
    let parsed_entries: Vec<MmmluDatasetEntryNormalized> = entries
        .into_iter()
        .map(|entry| serde_json::from_value(entry).expect("Failed to parse normalized MMMLU entry"))
        .collect();
    let mut correct_one_answer_entries: Vec<OneAnswerEntry> = Vec::new();
    let mut incorrect_one_answer_entries: Vec<OneAnswerEntry> = Vec::new();
    for entry in parsed_entries {
        let correct_answer_index = entry.answer;
        let incorrect_answer_index = (correct_answer_index + 1) % 4; // Just pick the next answer as incorrect
        correct_one_answer_entries.push(OneAnswerEntry {
            index: entry.original_index,
            question: entry.question.clone(),
            answer: entry.choices[correct_answer_index].clone(),
            lang: lang.to_string(),
            is_correct: true,
            subject: entry.subject.clone(),
        });
        incorrect_one_answer_entries.push(OneAnswerEntry {
            index: entry.original_index,
            question: entry.question.clone(),
            answer: entry.choices[incorrect_answer_index].clone(),
            lang: lang.to_string(),
            is_correct: false,
            subject: entry.subject.clone(),
        });
    }
    let correct_parsed: Vec<serde_json::Value> = correct_one_answer_entries
        .into_iter()
        .map(|entry| {
            serde_json::to_value(entry).expect("Failed to serialize correct one answer entry")
        })
        .collect();
    let incorrect_parsed: Vec<serde_json::Value> = incorrect_one_answer_entries
        .into_iter()
        .map(|entry| {
            serde_json::to_value(entry).expect("Failed to serialize incorrect one answer entry")
        })
        .collect();
    write_json_lines_to_file(&output_correct_path, &correct_parsed)
        .expect("Failed to write correct one answer dataset");
    write_json_lines_to_file(&output_incorrect_path, &incorrect_parsed)
        .expect("Failed to write incorrect one answer dataset");
}

pub fn generate_two_answers_dataset(lang1: &str, lang2: &str) {
    assert!(
        lang1 <= lang2,
        "lang1 should be less than or equal to lang2 to avoid duplicate generation"
    );
    let lang1_correct_lang2_incorrect_path = format!(
        "judge/datasets/two_answers/{}_correct_{}_incorrect.jsonl",
        lang1, lang2
    );
    let lang1_incorrect_lang2_correct_path = format!(
        "judge/datasets/two_answers/{}_incorrect_{}_correct.jsonl",
        lang1, lang2
    );
    let both_correct_path = format!(
        "judge/datasets/two_answers/{}_correct_{}_correct.jsonl",
        lang1, lang2
    );
    let both_incorrect_path = format!(
        "judge/datasets/two_answers/{}_incorrect_{}_incorrect.jsonl",
        lang1, lang2
    );
    let output_paths_exist = [
        &lang1_correct_lang2_incorrect_path,
        &lang1_incorrect_lang2_correct_path,
        &both_correct_path,
        &both_incorrect_path,
    ]
    .iter()
    .all(|path| Path::new(path).exists());
    if output_paths_exist {
        println!(
            "Two answers datasets for languages {} and {} already exist. Skipping generation.",
            lang1, lang2
        );
        return;
    }
    let input_path_lang1_correct = format!("judge/datasets/one_answer/{}_correct.jsonl", lang1);
    let input_path_lang1_incorrect = format!("judge/datasets/one_answer/{}_incorrect.jsonl", lang1);
    let input_path_lang2_correct = format!("judge/datasets/one_answer/{}_correct.jsonl", lang2);
    let input_path_lang2_incorrect = format!("judge/datasets/one_answer/{}_incorrect.jsonl", lang2);
    let input_paths_exist = [
        &input_path_lang1_correct,
        &input_path_lang1_incorrect,
        &input_path_lang2_correct,
        &input_path_lang2_incorrect,
    ]
    .iter()
    .all(|path| Path::new(path).exists());
    if !input_paths_exist {
        println!(
            "One answer datasets for languages {} and/or {} not found. Generating...",
            lang1, lang2
        );
        generate_one_answer_dataset(lang1);
        generate_one_answer_dataset(lang2);
    }
    let lang1_correct_entries =
        load_json_lines(&input_path_lang1_correct).expect("Failed to load lang1 correct dataset");
    let lang1_incorrect_entries = load_json_lines(&input_path_lang1_incorrect)
        .expect("Failed to load lang1 incorrect dataset");
    let lang2_correct_entries =
        load_json_lines(&input_path_lang2_correct).expect("Failed to load lang2 correct dataset");
    let lang2_incorrect_entries = load_json_lines(&input_path_lang2_incorrect)
        .expect("Failed to load lang2 incorrect dataset");
    let lang1_correct_entries: IndexMap<usize, OneAnswerEntry> = lang1_correct_entries
        .into_iter()
        .map(|entry| {
            let parsed: OneAnswerEntry =
                serde_json::from_value(entry).expect("Failed to parse lang1 correct entry");
            (parsed.index, parsed)
        })
        .collect();
    let lang1_incorrect_entries: IndexMap<usize, OneAnswerEntry> = lang1_incorrect_entries
        .into_iter()
        .map(|entry| {
            let parsed: OneAnswerEntry =
                serde_json::from_value(entry).expect("Failed to parse lang1 incorrect entry");
            (parsed.index, parsed)
        })
        .collect();
    let lang2_correct_entries: IndexMap<usize, OneAnswerEntry> = lang2_correct_entries
        .into_iter()
        .map(|entry| {
            let parsed: OneAnswerEntry =
                serde_json::from_value(entry).expect("Failed to parse lang2 correct entry");
            (parsed.index, parsed)
        })
        .collect();
    let lang2_incorrect_entries: IndexMap<usize, OneAnswerEntry> = lang2_incorrect_entries
        .into_iter()
        .map(|entry| {
            let parsed: OneAnswerEntry =
                serde_json::from_value(entry).expect("Failed to parse lang2 incorrect entry");
            (parsed.index, parsed)
        })
        .collect();
    let dataset_length = lang1_correct_entries.len();
    assert_eq!(dataset_length, lang1_incorrect_entries.len());
    assert_eq!(dataset_length, lang2_correct_entries.len());
    assert_eq!(dataset_length, lang2_incorrect_entries.len());
    let indices = lang1_correct_entries.keys();
    let mut lang1_correct_lang2_incorrect: Vec<TwoAnswersEntry> = Vec::new();
    let mut lang1_incorrect_lang2_correct: Vec<TwoAnswersEntry> = Vec::new();
    let mut both_correct: Vec<TwoAnswersEntry> = Vec::new();
    let mut both_incorrect: Vec<TwoAnswersEntry> = Vec::new();
    for index in indices {
        let entry_lang1_correct = lang1_correct_entries
            .get(index)
            .expect("Missing lang1 correct entry");
        let entry_lang1_incorrect = lang1_incorrect_entries
            .get(index)
            .expect("Missing lang1 incorrect entry");
        let entry_lang2_correct = lang2_correct_entries
            .get(index)
            .expect("Missing lang2 correct entry");
        let entry_lang2_incorrect = lang2_incorrect_entries
            .get(index)
            .expect("Missing lang2 incorrect entry");
        lang1_correct_lang2_incorrect.push(TwoAnswersEntry {
            index: *index,
            question: entry_lang1_correct.question.clone(),
            answer1: entry_lang1_correct.answer.clone(),
            answer2: entry_lang2_incorrect.answer.clone(),
            lang1: lang1.to_string(),
            lang2: lang2.to_string(),
            is_correct1: true,
            is_correct2: false,
            subject: entry_lang1_correct.subject.clone(),
        });
        lang1_incorrect_lang2_correct.push(TwoAnswersEntry {
            index: *index,
            question: entry_lang1_incorrect.question.clone(),
            answer1: entry_lang1_incorrect.answer.clone(),
            answer2: entry_lang2_correct.answer.clone(),
            lang1: lang1.to_string(),
            lang2: lang2.to_string(),
            is_correct1: false,
            is_correct2: true,
            subject: entry_lang1_incorrect.subject.clone(),
        });
        both_correct.push(TwoAnswersEntry {
            index: *index,
            question: entry_lang1_correct.question.clone(),
            answer1: entry_lang1_correct.answer.clone(),
            answer2: entry_lang2_correct.answer.clone(),
            lang1: lang1.to_string(),
            lang2: lang2.to_string(),
            is_correct1: true,
            is_correct2: true,
            subject: entry_lang1_correct.subject.clone(),
        });
        both_incorrect.push(TwoAnswersEntry {
            index: *index,
            question: entry_lang1_incorrect.question.clone(),
            answer1: entry_lang1_incorrect.answer.clone(),
            answer2: entry_lang2_incorrect.answer.clone(),
            lang1: lang1.to_string(),
            lang2: lang2.to_string(),
            is_correct1: false,
            is_correct2: false,
            subject: entry_lang1_incorrect.subject.clone(),
        });
    }
    let lang1_correct_lang2_incorrect_serialized: Vec<serde_json::Value> =
        lang1_correct_lang2_incorrect
            .into_iter()
            .map(|entry| {
                serde_json::to_value(entry)
                    .expect("Failed to serialize lang1 correct lang2 incorrect entry")
            })
            .collect();
    let lang1_incorrect_lang2_correct_serialized: Vec<serde_json::Value> =
        lang1_incorrect_lang2_correct
            .into_iter()
            .map(|entry| {
                serde_json::to_value(entry)
                    .expect("Failed to serialize lang1 incorrect lang2 correct entry")
            })
            .collect();
    let both_correct_serialized: Vec<serde_json::Value> = both_correct
        .into_iter()
        .map(|entry| serde_json::to_value(entry).expect("Failed to serialize both correct entry"))
        .collect();
    let both_incorrect_serialized: Vec<serde_json::Value> = both_incorrect
        .into_iter()
        .map(|entry| serde_json::to_value(entry).expect("Failed to serialize both incorrect entry"))
        .collect();
    write_json_lines_to_file(
        &lang1_correct_lang2_incorrect_path,
        &lang1_correct_lang2_incorrect_serialized,
    )
    .expect("Failed to write lang1 correct lang2 incorrect dataset");
    write_json_lines_to_file(
        &lang1_incorrect_lang2_correct_path,
        &lang1_incorrect_lang2_correct_serialized,
    )
    .expect("Failed to write lang1 incorrect lang2 correct dataset");
    write_json_lines_to_file(&both_correct_path, &both_correct_serialized)
        .expect("Failed to write both correct dataset");
    write_json_lines_to_file(&both_incorrect_path, &both_incorrect_serialized)
        .expect("Failed to write both incorrect dataset");
}

fn parse_and_normalize(raw_entry: &serde_json::Value, lang: &str) -> MmmluDatasetEntryNormalized {
    match lang {
        "en" => serde_json::from_value(raw_entry.clone())
            .expect("Failed to parse MMMLU dataset entry in English"),
        "zh_cn" => {
            let entry_chinese: MmmluDatasetEntryChinese = serde_json::from_value(raw_entry.clone())
                .expect("Failed to parse MMMLU dataset entry in Chinese");
            let answer = match entry_chinese.answer.as_str() {
                "A" => 0,
                "B" => 1,
                "C" => 2,
                "D" => 3,
                _ => panic!("Invalid answer choice: {}", entry_chinese.answer),
            };
            MmmluDatasetEntryNormalized {
                original_index: entry_chinese.original_index,
                question: entry_chinese.question,
                choices: [
                    entry_chinese.choice_a,
                    entry_chinese.choice_b,
                    entry_chinese.choice_c,
                    entry_chinese.choice_d,
                ],
                answer,
                subject: entry_chinese.subject,
            }
        }
        _ => panic!("Unsupported language: {}", lang),
    }
}

// pub fn generate_perplexity_dataset(lang: &str) {
//     let output_path = format!("judge/datasets/perplexity/{}.jsonl", lang);
//     if Path::new(&output_path).exists() {
//         println!(
//             "Perplexity dataset for language {} already exists. Skipping generation.",
//             lang
//         );
//         return;
//     }
//     let input_dataset_path = format!("judge/datasets/one_answer/{}_correct.jsonl", lang);
//     if !Path::new(&input_dataset_path).exists() {
//         println!(
//             "One answer dataset for language {} not found. Generating...",
//             lang
//         );
//         generate_one_answer_dataset(lang);
//     }
//     let entries =
//         load_json_lines(&input_dataset_path).expect("Failed to load one answer dataset");
//     let parsed_entries: Vec<SingleAnswerEntry> = entries
//         .into_iter()
//         .map(|entry| serde_json::from_value(entry).expect("Failed to parse one answer entry"))
//         .collect();
// }

pub fn generate_perplexity_dataset_mask() {
    Python::attach(|py| {
        let generate_perplexity_mask_module = py
            .import("src_py.judge.generate_perplexity_dataset_mask")
            .expect("Failed to import src_py.judge.generate_perplexity_dataset_mask module");
        let generate_perplexity_mask_func = generate_perplexity_mask_module
            .getattr("generate_perplexity_dataset_mask")
            .expect("Failed to get generate_perplexity_dataset_mask function");
        generate_perplexity_mask_func
            .call0()
            .expect("Failed to call generate_perplexity_dataset_mask function");
    });
    println!("Generated perplexity dataset mask.");
}

pub fn get_valid_perplexity_indices() -> HashSet<usize> {
    let perplexity_indices_path = "judge/datasets/valid_perplexity_indices.json";
    if !Path::new(&perplexity_indices_path).exists() {
        println!("Valid perplexity indices file not found. Generating...");
        generate_valid_perplexity_indices();
    }
    let file_content = std::fs::read_to_string(&perplexity_indices_path)
        .expect("Failed to read valid perplexity indices file");
    let valid_indices: HashSet<usize> =
        serde_json::from_str(&file_content).expect("Failed to parse valid perplexity indices");
    valid_indices
}

pub fn generate_valid_perplexity_indices() {
    let mask_dataset_path = "judge/datasets/perplexity_mask.jsonl";
    let output_path = "judge/datasets/valid_perplexity_indices.json";
    if !Path::new(&mask_dataset_path).exists() {
        println!("Perplexity dataset mask not found. Generating...");
        generate_perplexity_dataset_mask();
    }
    let mask_entries =
        load_json_lines(&mask_dataset_path).expect("Failed to load perplexity dataset mask");
    let valid_indices: Vec<usize> = mask_entries
        .into_iter()
        .filter_map(|entry| {
            let parsed = serde_json::from_value::<PerplexityDatasetMaskEntry>(entry)
                .expect("Failed to parse perplexity mask entry");
            if parsed.valid {
                Some(parsed.index)
            } else {
                None
            }
        })
        .collect();
    let serialized_indices =
        serde_json::to_string_pretty(&valid_indices).expect("Failed to serialize valid indices");
    std::fs::write(&output_path, &serialized_indices)
        .expect("Failed to write valid perplexity indices");
    println!("Generated valid perplexity indices.");
}

pub fn get_preference_indices() -> HashSet<usize> {
    let preference_indices_path = "judge/datasets/preference_indices.json";
    if !Path::new(&preference_indices_path).exists() {
        println!("Preference indices file not found. Generating...");
        generate_preference_indices();
    }
    let file_content = std::fs::read_to_string(&preference_indices_path)
        .expect("Failed to read preference indices file");
    let preference_indices: HashSet<usize> =
        serde_json::from_str(&file_content).expect("Failed to parse preference indices");
    preference_indices
}

pub fn generate_preference_indices() {
    let dataset_path = "judge/datasets/mmmlu_normalized/en.jsonl";
    let output_path = "judge/datasets/preference_indices.json";
    if !Path::new(&dataset_path).exists() {
        println!("Normalized MMMLU dataset for English not found. Generating...");
        generate_normalized_datasets("en");
    }
    let entries = load_json_lines(&dataset_path).expect("Failed to load normalized MMMLU dataset");
    let indices: Vec<usize> = entries
        .into_iter()
        .map(|entry| {
            let parsed = serde_json::from_value::<MmmluDatasetEntryNormalized>(entry)
                .expect("Failed to parse normalized MMMLU entry");
            parsed.original_index
        })
        .collect();
    let serialized_indices =
        serde_json::to_string_pretty(&indices).expect("Failed to serialize preference indices");
    std::fs::write(&output_path, &serialized_indices).expect("Failed to write preference indices");
    println!("Generated preference indices.");
}

#[test]
fn test_generate_normalized_dataset() {
    generate_normalized_datasets("en");
    generate_normalized_datasets("zh_cn");
}
