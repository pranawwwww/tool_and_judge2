use pyo3::{
    Bound,
    types::{PyAnyMethods, PyList, PyListMethods},
};
use tokio::runtime::Runtime;

use crate::{
    config::{Language, ToolConfig, TranslateMode, TranslateOption},
    models::{
        backend::{WhichBackend, get_or_create_backend},
        model_interface::get_model_interface,
    },
    tool_bfcl_decl::BfclDatasetEntry,
    util::{
        get_model_directory_safe_name, load_json_lines, load_json_lines_with_id,
        sort_and_write_json_lines, write_json_lines_to_file,
    },
};

const CATEGORY_CACHE_PATH: &str = "tool_category_cache.json";
const CATEGORY_CACHE_LOCK_PATH: &str = "tool_category_cache.json.lock";

pub fn tool_run_impl<'py>(configs: &Bound<'py, PyList>, num_gpus: usize) {
    println!(
        "Tool run implementation called with {} configs and {} GPUs.",
        configs.len(),
        num_gpus
    );
    let extracted_configs: Vec<ToolConfig> = configs
        .iter()
        .map(|config| {
            config
                .extract()
                .expect("Failed to extract ToolConfig from Python object")
        })
        .collect();

    // load environment variables from .env file
    dotenvy::dotenv().ok();
    let rt = Runtime::new().unwrap();
    rt.block_on(tool_run_async(extracted_configs, num_gpus));
}

pub async fn tool_run_async(configs: Vec<ToolConfig>, num_gpus: usize) {
    for config in configs {
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

        let test_cases =
            load_json_lines(&unpretranslated_dataset_path).expect("Failed to load test cases");
        let ground_truths =
            load_json_lines(ground_truth_path).expect("Failed to load ground truths");

        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* PASS 1: Translated Questions (Pre-Translation)                                   */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* Translates questions from the source language to English before inference.       */
        /* This pass runs when FULLY_TRANSLATED_PRE_TRANSLATE option is enabled.            */
        /* Output: tool/result/pre_translate/{model}/{language}.json                        */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        if pre_translate_tag == "_nopretrans" {
            // Skip translation - pass through original test cases
            println!("Skipping question translation (pre-translate not enabled)");
        } else {
            assert_eq!(pre_translate_tag, "_pretrans");
            let pre_translate_output_path = pre_translate_output_path
                .as_ref()
                .expect("pre_translate_output_path should have value");
            let (mut pre_translate_results, existing_pre_translate_ids) =
                match load_json_lines_with_id(&pre_translate_output_path) {
                    Ok(results) => results,
                    Err(_) => {
                        println!(
                            "File {} not found. It will be created.",
                            pre_translate_output_path
                        );
                        (Vec::new(), Vec::new())
                    }
                };
            let cases_to_translate: Vec<serde_json::Value> = test_cases
                .iter()
                .filter(|case| {
                    let case_id = case
                        .get("id")
                        .and_then(|id| id.as_str())
                        .expect("Case missing 'id' field");
                    !existing_pre_translate_ids.contains(&case_id.to_string())
                })
                .cloned()
                .collect();

            if cases_to_translate.is_empty() {
                println!("All test cases have already been translated. Skipping translation.");
            } else {
                println!(
                    "Translating {} questions to English...",
                    cases_to_translate.len()
                );

                let cases_to_translate_parsed: Vec<BfclDatasetEntry> = cases_to_translate
                    .iter()
                    .map(|case| {
                        BfclDatasetEntry::try_from(case.clone())
                            .expect("Dataset entry has wrong format")
                    })
                    .collect();

                // Get backend and interface for translation
                let main_backend =
                    get_or_create_backend(config.model, WhichBackend::Main, num_gpus).await;
                let main_backend = main_backend
                    .as_ref()
                    .expect("Backend should be created by the call above");
                let main_interface = get_model_interface(config.model);

                // async fn translate_single_question(
                //     case: &serde_json::Value,
                //     translation_backend: &crate::util::ModelBackend,
                //     translation_interface: &crate::util::ModelInterface,
                // ) -> serde_json::Value {
                let mut translate_single_question = async |case: &BfclDatasetEntry| {
                    let question = &case.question_content;
                    // Use the dedicated translation method
                    let translated_question = main_interface
                        .translate_tool_question_async(main_backend.as_ref(), question)
                        .await;
                    let modified_case = case
                        .modify_question_content(&translated_question)
                        .expect("Failed to modify question content");
                    modified_case
                };
                for (i, case) in cases_to_translate_parsed.iter().enumerate() {
                    let modified_case = translate_single_question(case).await;
                    println!(
                        "[{}/{}] Translated question for case {}",
                        i + 1,
                        cases_to_translate_parsed.len(),
                        modified_case
                            .get("id")
                            .and_then(|id| id.as_str())
                            .expect("Modified case missing 'id' field")
                    );
                    pre_translate_results.push(modified_case);
                    // Write to file immediately
                    write_json_lines_to_file(&pre_translate_output_path, &pre_translate_results)
                        .expect("Failed to write pre-translation results to file");
                }
                println!(
                    "All {} questions translated.",
                    cases_to_translate_parsed.len()
                );
                // Final sort and write
                if !pre_translate_results.is_empty() {
                    sort_and_write_json_lines(
                        pre_translate_output_path,
                        &mut pre_translate_results,
                    );
                }
            }
        }
    }
}

// # ═══════════════════════════════════════════════════════════════════════
//         # PASS 1: Translated Questions (Pre-Translation)
//         # ═══════════════════════════════════════════════════════════════════════
//         # Translates questions from the source language to English before inference.
//         # This pass runs when FULLY_TRANSLATED_PRE_TRANSLATE option is enabled.
//         # Output: tool/result/pre_translate/{model}/{language}.json
//         # ═══════════════════════════════════════════════════════════════════════
//         if pre_translate_tag == "_nopretrans":
//             # Skip translation - pass through original test cases
//             print(f"Skipping question translation (pre-translate not enabled)")
//         else:
//             assert pre_translate_tag == "_pretrans"
//             try:
//                 pre_translate_results, existing_pre_translate_ids = load_json_lines_with_id(pre_translate_output_path)
//                 existing_pre_translate_ids = {entry["id"] for entry in pre_translate_results}
//             except FileNotFoundError:
//                 print(f"File {pre_translate_output_path} not found. It will be created.")
//                 pre_translate_results = []
//                 existing_pre_translate_ids = set()

//             # Filter cases that haven't been translated yet
//             cases_to_translate = [case for case in test_cases if case['id'] not in existing_pre_translate_ids]

//             if len(cases_to_translate) == 0:
//                 print(f"All test cases have already been translated. Skipping translation.")
//             else:
//                 print(f"Translating {len(cases_to_translate)} questions to English...")

//                 # Get backend and interface for translation
//                 translation_backend = get_or_create_backend(
//                     model=config.model,
//                     num_gpus=args.num_gpus,
//                     max_model_len=2000,
//                     instance_name="experiment"  # Use experiment instance for pre-translation
//                 )
//                 translation_interface = get_or_create_model_interface(config.model)

//                 async def translate_questions_async():
//                     """Translate questions asynchronously."""
//                     async def translate_single_question(case):
//                         """Translate a single question and return the modified case."""
//                         question = case["question"][0][0]['content']

//                         # Use the dedicated translation method
//                         translated_question = await translation_interface.translate_tool_question_async(
//                             backend=translation_backend,
//                             question=question
//                         )

//                         # Create modified case with translated question
//                         modified_case = case.copy()
//                         modified_case["question"][0][0]['content'] = translated_question

//                         return modified_case

//                     # Create all translation tasks
//                     tasks = [translate_single_question(case) for case in cases_to_translate]

//                     # Process results as they complete
//                     completed_count = 0
//                     for coro in asyncio.as_completed(tasks):
//                         modified_case = await coro
//                         completed_count += 1

//                         print(f"[{completed_count}/{len(cases_to_translate)}] Translated question for case {modified_case['id']}")

//                         pre_translate_results.append(modified_case)

//                         # Write to file immediately
//                         write_json_lines_to_file(pre_translate_output_path, pre_translate_results)

//                 # Run the async translation
//                 await translate_questions_async()

//                 print(f"All {len(cases_to_translate)} questions translated.")

//                 # Final sort and write
//                 if len(pre_translate_results) > 0:
//                     append_and_rewrite_json_lines(pre_translate_output_path, pre_translate_results)
