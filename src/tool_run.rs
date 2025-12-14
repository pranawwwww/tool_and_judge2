use atomic_refcell::AtomicRefCell;
use futures::stream::{self, StreamExt};
use pyo3::{
    Py, Python,
    types::{PyAnyMethods, PyList, PyListMethods},
};
use std::{
    collections::HashMap,
    fs::{self, File},
    sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}},
};

use crate::{
    config::{Language, ToolConfig, TranslateMode, TranslateOption},
    models::{
        backend::{WhichBackend, get_or_create_backend},
        function_name_mapper::{self, FunctionNameMapper},
        model_interface::get_model_interface,
    },
    tool_bfcl_formats::{BfclDatasetEntry, BfclFunctionDef, BfclGroundTruthEntry},
    tool_categorize::categorize_entry,
    tool_category_cache::CategoryCache,
    tool_evaluate::evaluate_entry,
    tool_file_models::{
        CategorizedEntry, EvaluationResultEntry, EvaluationSummary, InferenceJsonEntry,
        InferenceRawEntry,
    },
    tool_translate_function_call::translate_function_call,
    util::{
        compare_id, deserialize_categorized_entries, deserialize_evaluation_result_entries,
        deserialize_ground_truth_entries, deserialize_inference_json_entries,
        deserialize_inference_raw_entries, deserialize_test_cases, get_model_directory_safe_name,
        load_json_lines, load_test_cases, serialize_categorized_entries,
        serialize_evaluation_result_entries, serialize_inference_json_entries,
        serialize_inference_raw_entries, serialize_test_cases, try_load_inference_json_and_ids,
        try_load_inference_raw_and_ids, try_load_test_cases_and_ids, write_json_lines_to_file,
    },
};

const CATEGORY_CACHE_PATH: &str = "tool_category_cache.jsonl";
const CATEGORY_CACHE_LOCK_PATH: &str = "tool_category_cache.lock";

// Maximum concurrent API requests. Adjust based on your OpenAI tier:
// - Tier 1: 500 RPM  -> use 20-30 concurrent
// - Tier 2: 5000 RPM -> use 50-80 concurrent
// - Tier 3+: higher  -> use 100+ concurrent
// Setting this too high will cause rate limit throttling (429 errors)
const MAX_CONCURRENT_REQUESTS: usize = 50;

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

    // Set up Ctrl+C handler for graceful shutdown
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        println!("\n⚠️  Ctrl+C detected! Finishing current task and shutting down...");
        shutdown_clone.store(true, Ordering::SeqCst);
    }).expect("Error setting Ctrl+C handler");

    let function_name_mapper = Arc::new(AtomicRefCell::new(FunctionNameMapper::new()));
    for config in extracted_configs {
        // Check for shutdown signal between configs
        if shutdown.load(Ordering::SeqCst) {
            println!("⚠️  Shutdown requested. Stopping before next config.");
            break;
        }

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
            "tool/dataset/BFCL_v4_multiple{language_tag}{translate_level_tag}{noise_tag}.jsonl"
        );
        let ground_truth_path = "tool/dataset/possible_answer/BFCL_v4_multiple.jsonl";
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
                    "tool/result/pre_translate/{model_dir_name}/{pre_translate_output_combined_tags}.jsonl"
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
            "tool/result/inference_raw/{model_dir_name}/{inference_raw_output_combined_tags}.jsonl"
        );

        let inference_json_input_path = inference_raw_output_path.clone();
        let inference_json_output_path = format!(
            "tool/result/inference_json/{model_dir_name}/{inference_raw_output_combined_tags}.jsonl"
        );
        let post_translate_input_path = inference_json_output_path.clone();
        let post_translate_output_path = if post_translate_tag == "_posttrans" {
            Some(format!(
                "tool/result/post_translate/{model_dir_name}/{post_translate_output_combined_tags}.jsonl"
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
            "tool/result/evaluation/{model_dir_name}/{post_translate_output_combined_tags}.jsonl"
        );
        let score_input_path = evaluation_output_path.clone();
        let score_output_path = format!(
            "tool/result/score/{model_dir_name}/{post_translate_output_combined_tags}.jsonl"
        );
        let categorize_input_path = score_output_path.clone();
        let categorize_output_path = format!(
            "tool/result/categorize/{model_dir_name}/{post_translate_output_combined_tags}.jsonl"
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

            // Create a stream from the tasks and process concurrently
            let mut translate_stream = stream::iter(tasks).buffer_unordered(MAX_CONCURRENT_REQUESTS);

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
            // Create a stream from the tasks and process concurrently
            let mut inference_stream = stream::iter(tasks).buffer_unordered(MAX_CONCURRENT_REQUESTS);
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
        let inference_raw_entries = deserialize_inference_raw_entries(inference_json_inputs);
        let main_interface = get_model_interface(config.model);
        let mut inference_json_outputs = Vec::new();
        // populate the name mapper with function names in the preprocessed dataset
        let preprocessed_test_cases = load_json_lines(&inference_raw_input_path)
            .expect("Failed to load pre-translation test cases for inference");
        let preprocessed_test_cases = deserialize_test_cases(preprocessed_test_cases);
        let mut all_functions: Vec<BfclFunctionDef> = Vec::new();
        for case in preprocessed_test_cases.iter() {
            for function in case.functions.iter() {
                all_functions.push(function.clone());
            }
        }
        {
            let mut fn_mapper = function_name_mapper.borrow_mut();
            fn_mapper.populate_from_functions(&all_functions);
        }
        for entry in inference_raw_entries.iter() {
            let id = entry.id.clone();
            let raw_output = &entry.raw_output;
            let result =
                main_interface.postprocess_tool_calls(raw_output, function_name_mapper.clone());
            let result = result.map(|func_calls| {
                func_calls
                    .into_iter()
                    .map(|func_call| func_call.serialize_to_json())
                    .collect::<Vec<serde_json::Value>>()
            });
            let valid = match &result {
                Ok(_) => true,
                Err(_) => false,
            };
            let output_entry = InferenceJsonEntry::new(id, valid, result);
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
            let inference_json_entries = deserialize_inference_json_entries(inference_json_inputs);

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
                        Ok(function_calls) => function_calls,
                        _ => {
                            return entry;
                        }
                    };
                    let mut translated_function_calls: HashMap<usize, serde_json::Value> =
                        HashMap::new();
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
                    let translated_function_calls: Vec<serde_json::Value> = (0
                        ..translated_function_calls.len())
                        .map(|i| {
                            translated_function_calls
                                .get(&i)
                                .cloned()
                                .expect("Translated function call should exist")
                        })
                        .collect();
                    InferenceJsonEntry::new(entry.id, entry.valid, Ok(translated_function_calls))
                };
                translate_functions_tasks.push(task);
            }
            let mut translate_stream =
                stream::iter(translate_functions_tasks).buffer_unordered(MAX_CONCURRENT_REQUESTS);
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
                translated_answers_results.sort_by(|a, b| compare_id(&a.id, &b.id));
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
        let inference_results = load_json_lines(&evaluation_input_path)
            .expect("Failed to load inference results for evaluation");
        let inference_results = deserialize_inference_json_entries(inference_results);
        let inference_results: HashMap<String, InferenceJsonEntry> = inference_results
            .into_iter()
            .map(|entry| (entry.id.clone(), entry))
            .collect();
        let test_cases = load_json_lines(&inference_raw_input_path)
            .expect("Failed to load test cases for evaluation");
        let test_cases = deserialize_test_cases(test_cases);
        let test_cases: HashMap<String, BfclDatasetEntry> = test_cases
            .into_iter()
            .map(|case| (case.id.clone(), case))
            .collect();
        let ground_truths = load_json_lines(ground_truth_path)
            .expect("Failed to load ground truths for evaluation");
        let ground_truths = deserialize_ground_truth_entries(ground_truths);
        let ground_truths: HashMap<String, BfclGroundTruthEntry> = ground_truths
            .into_iter()
            .map(|entry| (entry.id.clone(), entry))
            .collect();
        let mut evaluation_results: Vec<EvaluationResultEntry> = Vec::new();
        let ids: Vec<String> = inference_results.keys().cloned().collect();
        let total_cases = ids.len();
        println!("Evaluating {} cases...", total_cases);
        for (i, id) in ids.iter().enumerate() {
            let inference_result = inference_results
                .get(id)
                .expect("Inference result should exist");
            let test_case = test_cases.get(id).expect("Test case should exist");
            let ground_truth = ground_truths.get(id).expect("Ground truth should exist");
            let evaluation_result =
                evaluate_entry(id.into(), inference_result, test_case, ground_truth);
            
            evaluation_results.push(evaluation_result);
            println!("[{}/{}] Evaluated case {}", i + 1, total_cases, id);
        }
        // Final sort and write
        println!("Sorting evaluation results.");
        evaluation_results.sort_by(|a, b| compare_id(&a.id, &b.id));
        println!("Sorted evaluation results. Writing to file.");
        let evaluation_results_serialized =
            serialize_evaluation_result_entries(&evaluation_results);
        write_json_lines_to_file(&evaluation_output_path, &evaluation_results_serialized)
            .expect("Failed to sort and write evaluation results");
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* PASS 6: Score                                                          */
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* Calculates accuracy and aggregates wrong cases for analysis.          */
        /* Input: tool/result/evaluation/{model}/{filename}.json                 */
        /* Output: tool/result/score/{model}/{filename}.json                     */
        /* ═══════════════════════════════════════════════════════════════════════ */
        println!("Scoring evaluation results...");
        let evaluation_entries = load_json_lines(&score_input_path)
            .expect("Failed to load evaluation results for scoring");
        let evaluation_entries = deserialize_evaluation_result_entries(evaluation_entries);
        let mut total_cases = 0;
        let mut correct_cases = 0;
        let mut wrong_cases: Vec<EvaluationResultEntry> = Vec::new();
        for entry in evaluation_entries.iter() {
            total_cases += 1;
            if entry.valid {
                correct_cases += 1;
            } else {
                wrong_cases.push(entry.clone());
            }
        }
        let accuracy = if total_cases > 0 {
            correct_cases as f32 / total_cases as f32
        } else {
            0.0
        };
        let evaluation_summary = EvaluationSummary {
            accuracy,
            total_cases,
            correct_cases,
        };
        let evaluation_summary_json = serde_json::to_value(&evaluation_summary)
            .expect("Failed to serialize evaluation summary");
        let mut output_json_lines = serialize_evaluation_result_entries(&wrong_cases);
        output_json_lines.insert(0, evaluation_summary_json);
        write_json_lines_to_file(&score_output_path, &output_json_lines)
            .expect("Failed to write score results to file");
        println!(
            "Score result written to {}: {:?}",
            score_output_path, evaluation_summary
        );
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* PASS 7: Categorize                                                    */
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* Categorizes each error sample into different error types.            */
        /* Input: tool/result/score/{model}/{filename}.json                    */
        /* Output: tool/result/categorize/{model}/{filename}.json              */
        /* ═══════════════════════════════════════════════════════════════════════ */
        let categorize_pass = async || {
            let mut score_entries = load_json_lines(&categorize_input_path)
                .expect("Failed to load score results for categorization");
            // let evaluation_summary = &score_entries[0];
            score_entries.remove(0); // remove summary entry
            if score_entries.is_empty() {
                println!("No error samples found. Skipping categorization.");
                return;
            }
            println!("Categorizing {} error samples...", score_entries.len());
            let score_entries = deserialize_evaluation_result_entries(score_entries);

            println!("Acquiring lock for category cache file...");
            let lock_file = File::create(CATEGORY_CACHE_LOCK_PATH)
                .expect("Failed to create lock file for category cache");
            lock_file.lock().expect("Failed to lock the lock file");
            println!("Acquired lock for category cache file.");
            let category_cache = CategoryCache::load_or_create(CATEGORY_CACHE_PATH);
            let category_cache = Arc::new(AtomicRefCell::new(category_cache));
            let cache_hits = Arc::new(AtomicUsize::new(0));
            let main_interface = get_model_interface(config.model);
            let main_backend =
                get_or_create_backend(config.model, WhichBackend::Main, num_gpus).await;
            let main_backend = main_backend
                .as_ref()
                .expect("Backend should be created by the call above");

            let mut categorize_tasks = Vec::new();
            for entry in score_entries.iter() {
                let main_interface = main_interface.clone();
                let main_backend = main_backend.clone();
                let category_cache = category_cache.clone();
                let cache_hits = cache_hits.clone();

                let id = entry.id.clone();
                let error = entry
                    .error
                    .clone()
                    .expect("Error should exist for wrong cases");
                let task = async move {
                    let category = categorize_entry(
                        &error,
                        main_interface,
                        main_backend,
                        category_cache,
                        cache_hits,
                    )
                    .await;
                    CategorizedEntry {
                        id,
                        error_category: category,
                        error,
                    }
                };
                categorize_tasks.push(task);
            }
            let mut categorize_stream = stream::iter(categorize_tasks).buffer_unordered(MAX_CONCURRENT_REQUESTS);
            let mut categorized_entries: Vec<CategorizedEntry> = Vec::new();
            let mut completed_count = 0;
            while let Some(categorized_entry) = categorize_stream.next().await {
                completed_count += 1;
                println!(
                    "[{}/{}] Categorized error for case {}",
                    completed_count,
                    score_entries.len(),
                    categorized_entry.id
                );
                categorized_entries.push(categorized_entry);
            }
            println!("All {} error samples categorized.", score_entries.len());
            // Final sort and write
            categorized_entries.sort_by(|a, b| compare_id(&a.id, &b.id));
            let categorized_entries_serialized =
                serialize_categorized_entries(&categorized_entries);
            write_json_lines_to_file(&categorize_output_path, &categorized_entries_serialized)
                .expect("Failed to write to categorized file");
            // Save category cache
            let category_cache = category_cache.borrow();
            category_cache.save(CATEGORY_CACHE_PATH);
            // release the lock
            lock_file.unlock().expect("Failed to unlock the lock file");
            println!("Released lock for category cache file.");
        };
        categorize_pass().await;
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* PASS 8: Categorize Score                                              */
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* Aggregates categorization results and counts errors for each category.*/
        /* Lightweight pass that always overwrites existing file.               */
        /* Input: tool/result/categorize/{model}/{filename}.json                */
        /* Output: tool/result/categorize_score/{model}/{filename}.json        */
        /* ═══════════════════════════════════════════════════════════════════════ */
        let categorize_score_inputs = load_json_lines(&categorize_score_input_path)
            .expect("Failed to load categorize results for categorize scoring");
        let categorize_score_entries = deserialize_categorized_entries(categorize_score_inputs);
        // no need to skip because even if there is no categorized samples, we still want to write an empty summary
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        let mut category_samples: HashMap<String, Vec<String>> = HashMap::new();
        for entry in categorize_score_entries.iter() {
            let category = entry.error_category.to_string();
            *category_counts.entry(category.clone()).or_insert(0) += 1;
            category_samples
                .entry(category.clone())
                .or_insert_with(Vec::new)
                .push(entry.id.clone());
        }
        let final_output = serde_json::json!({
            "summary": category_counts,
            "samples": category_samples
        });
        let final_output_serialized = serde_json::to_string_pretty(&final_output)
            .expect("Failed to serialize categorize score output");
        // write the json object to file manually
        fs::create_dir_all(
            std::path::Path::new(&categorize_score_output_path)
                .parent()
                .expect("Failed to get parent directory for categorize score output path"),
        )
        .expect("Failed to create directories for categorize score output path");
        fs::write(categorize_score_output_path, final_output_serialized)
            .expect("Failed to write categorize score results to file");
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* All passes completed                                                    */
        /* ═══════════════════════════════════════════════════════════════════════ */
        println!("Completed processing for config: {:?}", config);
    }
}
