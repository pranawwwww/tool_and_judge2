use atomic_refcell::AtomicRefCell;
use futures::stream::{self, StreamExt};
use pyo3::{
    Py, Python,
    types::{PyAnyMethods, PyList, PyListMethods},
};
use std::{collections::HashMap, sync::Arc};

use crate::{
    config::{Language, ToolConfig, TranslateMode, TranslateOption}, models::{
        backend::{WhichBackend, get_or_create_backend},
        function_name_mapper::{self, FunctionNameMapper},
        model_interface::get_model_interface,
    }, tool_bfcl_decl::BfclDatasetEntry, tool_file_models::{InferenceJsonEntry, InferenceRawEntry, ToolCallParsingResult}, tool_translate_function_call::translate_function_call, util::{
        compare_id, deserialize_test_cases, get_model_directory_safe_name, load_json_lines,
        load_test_cases, parse_inference_json_entries, serialize_inference_json_entries,
        serialize_inference_raw_entries, serialize_test_cases, try_load_inference_json_and_ids,
        try_load_inference_raw_and_ids, try_load_test_cases_and_ids, write_json_lines_to_file,
    }
};

const CATEGORY_CACHE_PATH: &str = "tool_category_cache.json";
const CATEGORY_CACHE_LOCK_PATH: &str = "tool_category_cache.json.lock";

pub async fn tool_run_async(configs: Py<PyList>, num_gpus: usize) {
    let (extracted_configs, config_len): (Vec<ToolConfig>, usize) = Python::attach(|py| {
        let configs = configs.bind(py);
        let config_len = configs.len();
        let extracted_configs = configs
            .iter()
            .map(|config| {
                config
                    .extract()
                    .expect("Failed to extract ToolConfig from Python object")
            })
            .collect();
        (extracted_configs, config_len)
    });

    println!(
        "Tool run implementation called with {} configs and  {} GPUs.",
        config_len, num_gpus
    );

    // load environment variables from .env file
    dotenvy::dotenv().ok();
    println!("Loaded environment variables from .env file.");
    println!("Starting tool run with {} configs.", config_len);

    let function_name_mapper = Arc::new(AtomicRefCell::new(FunctionNameMapper::new()));
    for config in extracted_configs {
        println!("Processing config: {:?}", config);
        let language_tag = match &config.translate_mode {
            TranslateMode::Translated { language, .. } => match language {
                Language::Chinese => "_zh",
                Language::Hindi => "_hi",
            },
            TranslateMode::NotTranslated {} => "_en",
        };
        let (translate_level_tag, pre_translate_tag, prompt_translate_tag, post_translate_tag) =
            match &config.translate_mode {
                TranslateMode::Translated { option, .. } => match option {
                    TranslateOption::FullyTranslated => {
                        ("_fulltrans", "_nopretrans", "_noprompt", "_noposttrans")
                    }
                    TranslateOption::FullyTranslatedPromptTranslate => {
                        ("_fulltrans", "_nopretrans", "_prompt", "_noposttrans")
                    }
                    TranslateOption::FullyTranslatedPreTranslate => {
                        ("_fulltrans", "_pretrans", "_noprompt", "_noposttrans")
                    }
                    TranslateOption::FullyTranslatedPostTranslate => {
                        ("_fulltrans", "_nopretrans", "_noprompt", "_posttrans")
                    }
                    TranslateOption::PartiallyTranslated => {
                        ("_parttrans", "_nopretrans", "_noprompt", "_noposttrans")
                    }
                },
                TranslateMode::NotTranslated {} => {
                    ("_na", "_nopretrans", "_noprompt", "_noposttrans")
                }
            };
        let noise_tag = match &config.add_noise_mode {
            crate::config::AddNoiseMode::NoNoise => "_nonoise",
            crate::config::AddNoiseMode::Synonym => "_syno",
            crate::config::AddNoiseMode::Paraphrase => "_para",
        };
        let model_dir_name = get_model_directory_safe_name(&config.model.to_string());
        let unpretranslated_dataset_path = format!(
            "tool/dataset/BFCL_v4_multiple{language_tag}{translate_level_tag}{noise_tag}.json"
        );
        let ground_truth_path = "tool/dataset/possible_answer/BFCL_v4_multiple.json";
        let pre_translate_output_combined_tags =
            language_tag.to_string() + translate_level_tag + pre_translate_tag + noise_tag;
        let inference_raw_output_combined_tags = language_tag.to_string()
            + translate_level_tag
            + pre_translate_tag
            + noise_tag
            + prompt_translate_tag;
        let post_translate_output_combined_tags = language_tag.to_string()
            + translate_level_tag
            + pre_translate_tag
            + noise_tag
            + prompt_translate_tag
            + post_translate_tag;

        let (pre_translate_input_path, pre_translate_output_path) = if pre_translate_tag
            == "_pretrans"
        {
            (
                unpretranslated_dataset_path.clone(),
                Some(format!(
                    "tool/result/pre_translate/{model_dir_name}/{pre_translate_output_combined_tags}.json"
                )),
            )
        } else {
            assert_eq!(pre_translate_tag, "_nopretrans");
            (unpretranslated_dataset_path.clone(), None)
        };

        let inference_raw_input_path = if pre_translate_tag == "_pretrans" {
            pre_translate_output_path
                .clone()
                .expect("pre_translate_output_path should have value")
        } else {
            unpretranslated_dataset_path.clone()
        };

        let inference_raw_output_path = format!(
            "tool/result/inference_raw/{model_dir_name}/{inference_raw_output_combined_tags}.json"
        );

        let inference_json_input_path = inference_raw_output_path.clone();
        let inference_json_output_path = format!(
            "tool/result/inference_json/{model_dir_name}/{inference_raw_output_combined_tags}.json"
        );
        let post_translate_input_path = inference_json_output_path.clone();
        let post_translate_output_path = if post_translate_tag == "_posttrans" {
            Some(format!(
                "tool/result/post_translate/{model_dir_name}/{post_translate_output_combined_tags}.json"
            ))
        } else {
            assert_eq!(post_translate_tag, "_noposttrans");
            None
        };
        let evaluation_input_path = if post_translate_tag == "_posttrans" {
            post_translate_output_path
                .clone()
                .expect("post_translate_output_path should have value")
        } else {
            post_translate_input_path.clone()
        };
        let evaluation_output_path = format!(
            "tool/result/evaluation/{model_dir_name}/{post_translate_output_combined_tags}.json"
        );
        let score_input_path = evaluation_output_path.clone();
        let score_output_path = format!(
            "tool/result/score/{model_dir_name}/{post_translate_output_combined_tags}.json"
        );
        let categorize_input_path = score_output_path.clone();
        let categorize_output_path = format!(
            "tool/result/categorize/{model_dir_name}/{post_translate_output_combined_tags}.json"
        );
        let categorize_score_input_path = categorize_output_path.clone();
        let categorize_score_output_path = format!(
            "tool/result/categorize_score/{model_dir_name}/{post_translate_output_combined_tags}.json"
        );

        // let test_cases =
        //     load_json_lines(&unpretranslated_dataset_path).expect("Failed to load test cases");
        // let ground_truths =
        //     load_json_lines(ground_truth_path).expect("Failed to load ground truths");

        // let test_cases = parse_test_cases(test_cases);

        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* PASS 1: Translated Questions (Pre-Translation)                                   */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* Translates questions from the source language to English before inference.       */
        /* This pass runs when FULLY_TRANSLATED_PRE_TRANSLATE option is enabled.            */
        /* Output: tool/result/pre_translate/{model}/{language}.json                        */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        let pre_translate_pass = async || {
            if pre_translate_tag == "_nopretrans" {
                // Skip translation - pass through original test cases
                println!("Skipping question translation (pre-translate not enabled)");
                return;
            }
            assert_eq!(pre_translate_tag, "_pretrans");
            let pre_translate_output_path = pre_translate_output_path
                .as_ref()
                .expect("pre_translate_output_path should have value");
            let (mut pre_translate_results, existing_pre_translate_ids) =
                try_load_test_cases_and_ids(&pre_translate_output_path);
            let test_cases = load_test_cases(&pre_translate_input_path)
                .expect(&format!("Failed to open file {}", pre_translate_input_path));
            let cases_to_translate: Vec<BfclDatasetEntry> = test_cases
                .into_iter()
                .filter(|case| !existing_pre_translate_ids.contains(&case.id))
                .collect();

            if cases_to_translate.is_empty() {
                println!("All test cases have already been translated. Skipping translation.");
                return;
            }
            println!(
                "Translating {} questions to English...",
                cases_to_translate.len()
            );
            // Get backend and interface for translation
            let main_backend =
                get_or_create_backend(config.model, WhichBackend::Main, num_gpus).await;
            let main_backend = main_backend
                .as_ref()
                .expect("Backend should be created by the call above");
            let main_interface = get_model_interface(config.model);

            // Create tasks for all translations
            let total_cases = cases_to_translate.len();
            let mut tasks = Vec::new();

            for case in cases_to_translate.iter() {
                let question = case.question_content.clone();
                let mut case_clone = case.clone();
                let main_interface = main_interface.clone();
                let main_backend = main_backend.clone();
                let task = async move {
                    // Use the dedicated translation method
                    let translated_question = main_interface
                        .translate_tool_question_async(main_backend, question)
                        .await;
                    case_clone
                        .modify_question_content(&translated_question)
                        .expect("Failed to modify question content");
                    case_clone
                };
                tasks.push(task);
            }

            // Create a stream from the tasks and process up to 200 concurrently
            let mut translate_stream = stream::iter(tasks).buffer_unordered(200);

            let mut completed_count = 0;
            while let Some(modified_case) = translate_stream.next().await {
                completed_count += 1;
                println!(
                    "[{}/{}] Translated question for case {}",
                    completed_count, total_cases, modified_case.id
                );
                pre_translate_results.push(modified_case);
                // Write to file immediately
                if completed_count % 10 == 0 {
                    let serialized_test_cases = serialize_test_cases(&pre_translate_results);
                    write_json_lines_to_file(&pre_translate_output_path, &serialized_test_cases)
                        .expect("Failed to write pre-translation results to file");
                }
            }
            println!("All {} questions translated.", cases_to_translate.len());
            // Final sort and write
            if !pre_translate_results.is_empty() {
                pre_translate_results.sort_by(|a, b| compare_id(&a.id, &b.id));
                let serialized_test_cases = serialize_test_cases(&pre_translate_results);
                write_json_lines_to_file(&pre_translate_output_path, &serialized_test_cases)
                    .expect("Failed to write pre-translation results to file");
            }
        };
        pre_translate_pass().await;
        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* PASS 2: Inference Raw                                                            */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* Generates raw model outputs for each test case using function calling.           */
        /* Input: test_cases (from pre_translate if pre-translate enabled, else dataset)    */
        /* Output: tool/result/inference_raw/{model}/{filename}.json                        */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        let inference_raw_pass = async || {
            let (mut inference_raw_outputs, existing_inference_ids) =
                try_load_inference_raw_and_ids(&inference_raw_output_path);

            let preprocessed_test_cases = load_json_lines(&inference_raw_input_path)
                .expect("Failed to load pre-translation test cases for inference");
            let preprocessed_test_cases = deserialize_test_cases(preprocessed_test_cases);
            let cases_to_process = preprocessed_test_cases
                .into_iter()
                .filter(|case| !existing_inference_ids.contains(&case.id))
                .collect::<Vec<BfclDatasetEntry>>();
            if cases_to_process.is_empty() {
                println!(
                    "All test cases for {} have already been processed. Skipping model loading and inference.",
                    config.model.to_string()
                );
                return;
            }
            let total_cases = cases_to_process.len();
            println!("Generating functions for {} cases...", total_cases);
            let main_backend =
                get_or_create_backend(config.model, WhichBackend::Main, num_gpus).await;
            let main_backend = main_backend
                .as_ref()
                .expect("Backend should be created by the call above");
            // Model interface can be created outside async context
            let main_interface = get_model_interface(config.model);

            let prompt_translate = if prompt_translate_tag == "_prompt" {
                true
            } else {
                assert_eq!(prompt_translate_tag, "_noprompt");
                false
            };
            let mut tasks = Vec::new();

            for case in cases_to_process.iter() {
                let functions = case.functions.clone();
                let user_question = case.question_content.clone();
                let case_id = case.id.clone();
                let main_interface = main_interface.clone();
                let main_backend = main_backend.clone();
                let function_name_mapper = function_name_mapper.clone();
                let task = async move {
                    let result = main_interface
                        .generate_tool_call_async(
                            main_backend.clone(),
                            functions,
                            user_question,
                            prompt_translate,
                            function_name_mapper,
                        )
                        .await;
                    InferenceRawEntry::new(case_id, result)
                };
                tasks.push(task);
            }
            // Create a stream from the tasks and process up to 200 concurrently
            let mut inference_stream = stream::iter(tasks).buffer_unordered(200);
            let mut completed_count = 0;
            while let Some(result) = inference_stream.next().await {
                completed_count += 1;
                println!(
                    "[{}/{}] Case {} processed",
                    completed_count, total_cases, result.id
                );
                inference_raw_outputs.push(result);
                // Write to file immediately
                if completed_count % 10 == 0 {
                    let inference_raw_outputs_serialized =
                        serialize_inference_raw_entries(&inference_raw_outputs);
                    write_json_lines_to_file(
                        &inference_raw_output_path,
                        &inference_raw_outputs_serialized,
                    )
                    .expect("Failed to write inference raw results to file");
                }
            }
            println!("All {} cases processed.", cases_to_process.len());
            // Final sort and write
            if !inference_raw_outputs.is_empty() {
                inference_raw_outputs.sort_by(|a, b| compare_id(&a.id, &b.id));
                let inference_raw_outputs_serialized =
                    serialize_inference_raw_entries(&inference_raw_outputs);
                write_json_lines_to_file(
                    &inference_raw_output_path,
                    &inference_raw_outputs_serialized,
                )
                .expect("Failed to sort and write inference raw results");
            }
        };
        inference_raw_pass().await;
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* PASS 3: Inference JSON                                                  */
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* Converts raw model outputs into structured JSON format.                 */
        /* Input: tool/result/inference_raw/{model}/{filename}.json                */
        /* Output: tool/result/inference_json/{model}/{filename}.json              */
        /* ═══════════════════════════════════════════════════════════════════════ */
        let inference_json_inputs = load_json_lines(&inference_json_input_path)
            .expect("Failed to load inference raw outputs for JSON conversion");
        let main_interface = get_model_interface(config.model);
        let mut inference_json_outputs = Vec::new();
        for entry in inference_json_inputs.iter() {
            let id = entry
                .get("id")
                .and_then(|v| v.as_str())
                .expect("Missing or invalid 'id' field")
                .to_string();
            let result_str = entry
                .get("result")
                .and_then(|v| v.as_str())
                .expect("Missing or invalid 'result' field")
                .to_string();
            let result =
                main_interface.postprocess_tool_calls(&result_str, function_name_mapper.clone());
            let valid = match &result {
                ToolCallParsingResult::Success(_) => true,
                ToolCallParsingResult::Failure(_) => false,
            };
            let output_entry = InferenceJsonEntry::new(id, valid, result);
            // let output_entry = serde_json::to_value(output_entry)
            //     .expect("Failed to serialize InferenceJsonEntry to JSON value");
            inference_json_outputs.push(output_entry);
        }
        // Final sort and write
        inference_json_outputs.sort_by(|a, b| compare_id(&a.id, &b.id));
        let inference_json_outputs_serialized =
            serialize_inference_json_entries(&inference_json_outputs);
        write_json_lines_to_file(
            &inference_json_output_path,
            &inference_json_outputs_serialized,
        )
        .expect("Failed to sort and write inference JSON results");
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* PASS 4: Post-Translation                                                */
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* Translates model outputs back to the source language.                   */
        /* This pass runs when FULLY_TRANSLATED_POST_TRANSLATE option is enabled.  */
        /* Output: tool/result/post_translate/{model}/{language}.json              */
        /* ═══════════════════════════════════════════════════════════════════════ */
        let post_translate_pass = async || {
            if post_translate_tag == "_noposttrans" {
                // Skip translation - pass through original inference json results
                println!("Skipping answer translation (post-translate not enabled)");
                return;
            }
            assert_eq!(post_translate_tag, "_posttrans");
            let post_translate_output_path = post_translate_output_path
                .as_ref()
                .expect("post_translate_output_path should have value");
            // Load inference json results
            let inference_json_inputs = load_json_lines(&post_translate_input_path)
                .expect("Failed to load inference JSON results for post-translation");
            let inference_json_entries = parse_inference_json_entries(inference_json_inputs);

            let (mut translated_answers_results, existing_translated_answers_ids) =
                try_load_inference_json_and_ids(&post_translate_output_path);
            let samples_to_translate: Vec<InferenceJsonEntry> = inference_json_entries
                .into_iter()
                .filter(|entry| !existing_translated_answers_ids.contains(&entry.id))
                .collect();
            if samples_to_translate.is_empty() {
                println!("All answers have already been translated. Skipping translation.");
                return;
            }
            println!(
                "Translating {} answers to English...",
                samples_to_translate.len()
            );
            let main_backend =
                get_or_create_backend(config.model, WhichBackend::Main, num_gpus).await;
            let main_backend = main_backend
                .as_ref()
                .expect("Backend should be created by the call above");
            let main_interface = get_model_interface(config.model);
            let total_cases = samples_to_translate.len();
            let mut translate_functions_tasks = Vec::new();
            for entry in samples_to_translate.iter() {
                let entry = entry.clone();
                let main_interface = main_interface.clone();
                let main_backend = main_backend.clone();
                let task = async move {
                    let function_calls = match entry.result {
                        ToolCallParsingResult::Success(function_calls) => function_calls,
                        _ => {
                            return entry;
                        }
                    };
                    let mut translated_function_calls: HashMap<usize, serde_json::Value> = HashMap::new();
                    let mut translate_single_function_tasks = Vec::new();
                    for (i, func_call) in function_calls.iter().enumerate() {
                        let main_interface = main_interface.clone();
                        let main_backend = main_backend.clone();
                        let task = async move {
                            let translated_function_call = translate_function_call(
                                main_interface.clone(),
                                main_backend.clone(),
                                func_call.clone(),
                            )
                            .await;
                        (i, translated_function_call)
                        };
                        translate_single_function_tasks.push(task);
                    }
                    let results = futures::future::join_all(translate_single_function_tasks).await;
                    for (i, translated_function_call) in results {
                        translated_function_calls.insert(i, translated_function_call);
                    }
                    // reorder
                    let translated_function_calls: Vec<serde_json::Value> = (0..translated_function_calls.len())
                        .map(|i| {
                            translated_function_calls
                                .get(&i)
                                .cloned()
                                .expect("Translated function call should exist")
                        }).collect();
                    InferenceJsonEntry{
                        id: entry.id,
                        valid: entry.valid,
                        result: ToolCallParsingResult::Success(translated_function_calls),
                    }
                };
                translate_functions_tasks.push(task);                
            }
            let mut translate_stream = stream::iter(translate_functions_tasks).buffer_unordered(200);
            let mut completed_count = 0;
            while let Some(translated_entry) = translate_stream.next().await {
                completed_count += 1;
                println!(
                    "[{}/{}] Translated answer for case {}",
                    completed_count, total_cases, translated_entry.id
                );
                translated_answers_results.push(translated_entry);
                // Write to file immediately
                if completed_count % 10 == 0 {
                    let serialized_translated_answers =
                        serialize_inference_json_entries(&translated_answers_results);
                    write_json_lines_to_file(
                        &post_translate_output_path,
                        &serialized_translated_answers,
                    )
                    .expect("Failed to write translated answers to file");
                }
            }
            println!("All {} answers translated.", samples_to_translate.len());
            // Final sort and write
            if !translated_answers_results.is_empty() {
                translated_answers_results
                    .sort_by(|a, b| compare_id(&a.id, &b.id));
                let serialized_translated_answers =
                    serialize_inference_json_entries(&translated_answers_results);
                write_json_lines_to_file(
                    &post_translate_output_path,
                    &serialized_translated_answers,
                )
                .expect("Failed to write translated answers to file");
            }
        };
        post_translate_pass().await;
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* PASS 5: Evaluation                                                      */       
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* Evaluates model outputs against ground truth to determine correctness.  */
        /* Checks function names, parameter names, and parameter values.           */
        /* Input: tool/result/post_translate if post-translate enabled, else inference_json */
        /* Output: tool/result/evaluation/{model}/{filename}.json                  */
        /* ═══════════════════════════════════════════════════════════════════════ */
        println!("PASS 5: Evaluation not yet implemented.");
    }
}







// # ═══════════════════════════════════════════════════════════════════════
//         # PASS 5: Evaluation
//         # ═══════════════════════════════════════════════════════════════════════
//         # Evaluates model outputs against ground truth to determine correctness.
//         # Checks function names, parameter names, and parameter values.
//         # Input: tool/result/post_translate if post-translate enabled, else inference_json
//         # Output: tool/result/evaluation/{model}/{filename}.json
//         # ═══════════════════════════════════════════════════════════════════════
//         # reload inference results for evaluation
//         try:
//             inference_results = load_json_lines(evaluation_input_path)
//         except FileNotFoundError:
//             print(f"File {evaluation_input_path} not found. Skipping evaluation.")
//             exit(1)
//         evaluation_results = []

//         for (inference_line, ground_truth_line, test_case) in zip(inference_results, ground_truths, test_cases):
//             id = inference_line["id"]
//             assert id == ground_truth_line["id"], f"Mismatch in IDs: {id} vs {ground_truth_line['id']}"
//             assert id == test_case["id"], f"Mismatch in IDs: {id} vs {test_case['id']}"

//             # Check if postprocess result was valid
//             if inference_line.get("valid", True):  # Default to True for backward compatibility
//                 # Valid result: evaluate normally
//                 inference_result = inference_line["result"]
//                 ground_truth = ground_truth_line["ground_truth"]
//                 func_description = test_case['function']

//                 eval_result = evaluate_json(id, inference_result, ground_truth, func_description)

//                 # Check if evaluation returned error or success
//                 if isinstance(eval_result, tuple):
//                     # Evaluation error
//                     error, metadata = eval_result
//                     evaluation_entry = {
//                         "id": id,
//                         "valid": False,
//                         "error": error.value,
//                         "error_meta": metadata
//                     }
//                 else:
//                     # Evaluation success
//                     assert isinstance(eval_result, dict)
//                     evaluation_entry = eval_result
//             else:
//                 # Invalid result from postprocess: pass through the error
//                 evaluation_entry = {
//                     "id": id,
//                     "valid": False,
//                     "error": inference_line.get('error', 'unknown'),
//                     "error_meta": inference_line.get("error_meta", {})
//                 }

//             evaluation_results.append(evaluation_entry)

//             # Write batch results to file
//             write_json_lines_to_file(evaluation_output_path, evaluation_results)

//         # Final sort and write
//         if len(evaluation_results) > 0:
//             append_and_rewrite_json_lines(evaluation_output_path, evaluation_results)
//         # ═══════════════════════════════════════════════════════════════════════
//         # PASS 6: Score
//         # ═══════════════════════════════════════════════════════════════════════
//         # Calculates accuracy and aggregates wrong cases for analysis.
//         # Input: tool/result/evaluation/{model}/{filename}.json
//         # Output: tool/result/score/{model}/{filename}.json
//         # ═══════════════════════════════════════════════════════════════════════
//         # reload evaluation results
//         try:
//             evaluation_entries = load_json_lines(score_input_path)
//         except FileNotFoundError:
//             print(f"File {score_input_path} not found. Skipping scoring.")
//             exit(1)
//         # Calculate and write score results
//         total_cases = 0
//         correct_cases = 0
//         wrong_cases = []
//         score_results = []

//         for evaluation_entry in evaluation_entries:
//             total_cases += 1
//             if evaluation_entry['valid']:
//                 correct_cases += 1
//             else:
//                 wrong_cases.append(evaluation_entry)

//         accuracy = correct_cases / total_cases if total_cases > 0 else 0.0
//         # Add summary score
//         score_result = {
//             "accuracy": accuracy,
//             "total_cases": total_cases,
//             "correct_cases": correct_cases,
//         }
//         score_results.append(score_result)

//         # Add wrong cases
//         score_results.extend(wrong_cases)

//         # Write all results to file
//         write_json_lines_to_file(score_output_path, score_results)
//         print(f"Score result written to {score_output_path}: {score_result}")

//         # ═══════════════════════════════════════════════════════════════════════
//         # PASS 7: Categorize
//         # ═══════════════════════════════════════════════════════════════════════
//         # Categorizes each error sample into different error types.
//         # Input: tool/result/score/{model}/{filename}.json
//         # Output: tool/result/categorize/{model}/{filename}.json
//         # ═══════════════════════════════════════════════════════════════════════
//         # Load error samples from score file
//         try:
//             score_entries = load_json_lines(categorize_input_path)
//         except FileNotFoundError:
//             print(f"File {categorize_input_path} not found. Skipping categorization.")
//             score_entries = []

//         # Filter out the summary entry (first line in score file)
//         samples_to_categorize = [entry for entry in score_entries if 'accuracy' not in entry]

//         if len(samples_to_categorize) == 0:
//             print(f"No error samples found. Skipping categorization.")
//         else:
//             print(f"Categorizing {len(samples_to_categorize)} error samples...")

//             # Acquire cache lock for entire categorize pass
//             print("Acquiring cache lock...")
//             cache_lock.acquire()
//             print("Acquired cache lock")

//             # Load category cache
//             category_cache = load_category_cache(category_cache_path)

//             categorize_results = []
//             cache_hits = 0
//             cache_misses = 0

//             async def categorize_samples_async():
//                 """Categorize error samples asynchronously."""
//                 nonlocal cache_hits, cache_misses

//                 async def categorize_with_sample(sample):
//                     """Wrapper to return sample, category, and cache hit status."""
//                     category_enum, cache_hit = await categorize_single_sample_async(sample, category_cache)
//                     return sample, category_enum, cache_hit

//                 # Create all categorization tasks
//                 tasks = [categorize_with_sample(sample) for sample in samples_to_categorize]

//                 # Process results as they complete
//                 completed_count = 0
//                 for coro in asyncio.as_completed(tasks):
//                     sample, category_enum, cache_hit = await coro
//                     completed_count += 1

//                     # Track cache statistics
//                     if cache_hit:
//                         cache_hits += 1
//                     else:
//                         cache_misses += 1

//                     # Assemble the result dict (assembly logic in tool_main.py)
//                     categorized_sample = {
//                         "id": sample["id"],
//                         "category": category_enum.value,  # Store enum value as string
//                         "evaluation_entry": sample
//                     }

//                     # Only print on cache miss, and only show parameter comparison details
//                     if not cache_hit:
//                         error_meta = sample.get("error_meta", {})
//                         actual_value = error_meta.get("actual_value")
//                         expected_values = error_meta.get("expected_values", [])
//                         if actual_value is not None and expected_values:
//                             print(f"[{completed_count}/{len(samples_to_categorize)}] Comparing actual: {json.dumps(actual_value, ensure_ascii=False)} with expected: {json.dumps(expected_values, ensure_ascii=False)} -> {category_enum.value}")

//                     categorize_results.append(categorized_sample)

//                     # Write to file immediately
//                     write_json_lines_to_file(categorize_output_path, categorize_results)

//             try:
//                 # Run the async categorization
//                 await categorize_samples_async()

//                 print(f"All {len(samples_to_categorize)} samples categorized.")
//                 print(f"Cache statistics - Hits: {cache_hits}, Misses: {cache_misses}, Hit rate: {cache_hits / (cache_hits + cache_misses) * 100:.2f}%" if (cache_hits + cache_misses) > 0 else "Cache statistics - No cache lookups performed")

//                 # Final sort and write
//                 if len(categorize_results) > 0:
//                     append_and_rewrite_json_lines(categorize_output_path, categorize_results)

//                 # Write cache back to file once at the end
//                 save_category_cache(category_cache_path, category_cache)
//             finally:
//                 # Release lock at the end of categorize pass
//                 cache_lock.release()
//                 print("Released cache lock")

//             # Destroy local cache container (Python will handle this automatically when it goes out of scope)
//             del category_cache