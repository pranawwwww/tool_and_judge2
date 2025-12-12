use std::{collections::HashMap, sync::Arc};

use futures::{StreamExt, stream};
use serde_json::Map;

use crate::models::{backend::ModelBackend, model_interface::ModelInterface};

pub async fn translate_function_call(
    model_interface: Arc<dyn ModelInterface>,
    backend: Arc<dyn ModelBackend>,
    function: serde_json::Value,
) -> serde_json::Value {
    let function = function
        .as_object()
        .expect("Function call should be a JSON object");
    let function_name = function
        .keys()
        .next()
        .expect("Function call should have a function name");
    let params = function
        .get(function_name)
        .expect("Function call should have parameters");
    let params = params
        .as_object()
        .expect("Function parameters should be a JSON object");
    // let translated_params = params.iter().map(|(key, value)| {
    //     let translated_value = translate_param_value(value.clone()).await;
    //     (key.clone(), translated_value)
    // }).collect::<serde_json::Map<String, serde_json::Value>>();

    let mut translated_params = serde_json::Map::new();
    let mut tasks = Vec::new();
    for (key, value) in params {
        let key = key.clone();
        let value = value.clone();
        println!(
            "Submitted param value translation task for value: {}",
            value
        );
        let model_interface = model_interface.clone();
        let backend = backend.clone();
        let task = async move {
            let translated_value = translate_param_value(&value, model_interface, backend).await;
            (key, translated_value)
        };
        tasks.push(task);
    }
    let mut stream = stream::iter(tasks).buffer_unordered(200);
    while let Some((key, translated_value)) = stream.next().await {
        translated_params.insert(key, translated_value);
    }
    // make the translated params to have the same order as the original params
    let translated_params = params
        .iter()
        .map(|(key, value)| {
            (
                key.clone(),
                translated_params
                    .get(key)
                    .cloned()
                    .expect("Translated value should exist"),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();

    let mut new_function = function.clone();
    new_function.insert(
        function_name.clone(),
        serde_json::Value::Object(translated_params),
    );
    serde_json::Value::Object(new_function)
}

pub async fn translate_param_value(
    value: &serde_json::Value,
    model_interface: Arc<dyn ModelInterface>,
    backend: Arc<dyn ModelBackend>,
) -> serde_json::Value {
    match &value {
        serde_json::Value::Array(array) => {
            println!("Found an array as param value to translate: {}", value);
            let mut translated_array: HashMap<usize, serde_json::Value> = HashMap::new();
            let mut tasks = Vec::new();
            for (i, item) in array.into_iter().enumerate() {
                let model_interface = model_interface.clone();
                let backend = backend.clone();
                let task = async move {
                    let translated_item =
                        translate_param_value(&item, model_interface, backend).await;
                    (i, translated_item)
                };
                tasks.push(task);
            }
            let mut stream = stream::iter(tasks).buffer_unordered(200);
            while let Some((i, translated_item)) = stream.next().await {
                translated_array.insert(i, translated_item);
            }
            // reorder
            let translated_array = (0..translated_array.len())
                .map(|i| {
                    translated_array
                        .get(&i)
                        .cloned()
                        .expect("Translated array item should exist")
                })
                .collect::<Vec<serde_json::Value>>();
            serde_json::Value::Array(translated_array)
        }
        serde_json::Value::Object(obj) => {
            println!("Found an object as param value to translate: {}", value);
            let mut translated_obj: HashMap<usize, (String, serde_json::Value)> = HashMap::new();
            let mut tasks = Vec::new();
            for (i, (key, val)) in obj.into_iter().enumerate() {
                // translate both key and value
                let key = key.clone();
                let val = val.clone();
                let model_interface = model_interface.clone();
                let backend = backend.clone();
                let task = async move {
                    let translated_key = model_interface
                        .translate_tool_answer_async(backend.clone(), key)
                        .await;
                    let translated_value =
                        translate_param_value(&val, model_interface, backend).await;
                    (i, (translated_key, translated_value))
                };
                tasks.push(task);
            }
            let mut stream = stream::iter(tasks).buffer_unordered(200);
            while let Some((i, (translated_key, translated_value))) = stream.next().await {
                translated_obj.insert(i, (translated_key, translated_value));
            }
            // reorder
            let new_obj: Map<String, serde_json::Value> = (0..translated_obj.len())
                .map(|i| {
                    translated_obj
                        .get(&i)
                        .cloned()
                        .expect("Translated object item should exist")
                })
                .collect();
            serde_json::Value::Object(new_obj)
        }
        serde_json::Value::String(s) => {
            let translated_str = model_interface
                .translate_tool_answer_async(backend.clone(), s.clone())
                .await;
            serde_json::Value::String(translated_str)
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            value.clone()
        }
    }
}

// sample function call:
// {"triangle_properties.get": {"side1": 5, "side2": 4, "side3": 3, "get_area": true, "get_perimeter": true, "get_angles": true}}
