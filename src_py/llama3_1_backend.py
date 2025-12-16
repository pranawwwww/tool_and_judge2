"""
Llama 3.1 backend implementation using vLLM.

This backend implements Llama 3.1's tool calling format as specified in:
https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct#tool-use-with-transformers
"""

import json
from typing import Any, List


async def generate_tool_call_async(
    model_name: str,
    engine: Any,
    tokenizer: Any,
    question: str,
    tools: List[dict],
    prompt_passing_in_english: bool
) -> str:
    """
    Generate tool calls using Llama 3.1's native tool calling format.

    Args:
        model_name: The model name
        engine: vLLM AsyncLLMEngine instance
        tokenizer: Tokenizer instance
        question: User question
        tools: List of tool definitions in Llama 3.1 format
        prompt_passing_in_english: Whether to pass parameters in English

    Returns:
        JSON string containing the tool calls
    """
    # Build messages for Llama 3.1's chat template
    system_message = {
        "role": "system",
        "content": (
            "You are an expert in composing functions. "
            "You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
            "If none of the functions can be used, point it out. "
            "If the given question lacks the parameters required by the function, also point it out.\n\n"
            "You should ONLY return function calls in your response. "
            "You MUST NOT include any other text, explanations, or direct answers. "
            "If you decide to invoke any function(s), you MUST use the provided tools. "
            "Do NOT attempt to answer the question directly without using the available functions."
            f"{' IMPORTANT: Pass all parameter values in English' if prompt_passing_in_english else ''}"
        )
    }

    messages = [
        system_message,
        {"role": "user", "content": question}
    ]

    # Apply chat template with tools
    # The tokenizer.apply_chat_template will format the prompt according to Llama 3.1's conventions
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False
    )

    # Use vLLM to generate the response
    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for tool calls
        max_tokens=2048,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    # Generate with vLLM engine
    request_id = f"llama31_toolcall_{id(question)}"
    results_generator = engine.generate(
        formatted_prompt,
        sampling_params,
        request_id
    )

    # Wait for completion
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    # Extract the generated text
    generated_text = final_output.outputs[0].text.strip()

    # Parse the tool calls from the generated text
    # Llama 3.1 outputs tool calls in a specific format that needs to be extracted
    tool_calls = parse_llama3_1_tool_calls(generated_text)

    # Convert to the expected output format
    response_dicts = []
    for tool_call in tool_calls:
        response_dicts.append({
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["arguments"])
            }
        })

    response_json_str = json.dumps(response_dicts)
    return response_json_str


async def translate_tool_question_async(
    model_name: str,
    engine: Any,
    tokenizer: Any,
    question: str
) -> str:
    """
    Translate a question to English using Llama 3.1.

    Args:
        model_name: The model name
        engine: vLLM AsyncLLMEngine instance
        tokenizer: Tokenizer instance
        question: Question to translate

    Returns:
        Translated question in English
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
        },
        {
            "role": "user",
            "content": f"Translate the following question to English. Only output the translated question, nothing else:\n\n{question}"
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    request_id = f"llama31_translate_q_{id(question)}"
    results_generator = engine.generate(
        formatted_prompt,
        sampling_params,
        request_id
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    return final_output.outputs[0].text.strip()


async def translate_tool_answer_async(
    model_name: str,
    engine: Any,
    tokenizer: Any,
    parameter_value: str
) -> str:
    """
    Translate a parameter value to English using Llama 3.1.

    Args:
        model_name: The model name
        engine: vLLM AsyncLLMEngine instance
        tokenizer: Tokenizer instance
        parameter_value: Parameter value to translate

    Returns:
        Translated parameter value in English
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
        },
        {
            "role": "user",
            "content": f"Translate the following text to English. Only output the translated text, nothing else:\n\n{parameter_value}"
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    request_id = f"llama31_translate_a_{id(parameter_value)}"
    results_generator = engine.generate(
        formatted_prompt,
        sampling_params,
        request_id
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    return final_output.outputs[0].text.strip()


def parse_llama3_1_tool_calls(generated_text: str) -> List[dict]:
    """
    Parse tool calls from Llama 3.1's generated text.

    Llama 3.1 outputs tool calls in a structured format. This function extracts them.

    Args:
        generated_text: The generated text from the model

    Returns:
        List of tool call dictionaries with 'name' and 'arguments' keys
    """
    tool_calls = []

    # Try to parse as JSON first (if model outputs JSON directly)
    try:
        parsed = json.loads(generated_text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Look for tool call patterns in the text
    # Llama 3.1 typically outputs: <function=function_name>{"arg1": "value1", ...}</function>
    # or other structured formats
    import re

    # Pattern 1: <function=name>{...}</function>
    pattern1 = r'<function=([^>]+)>(.*?)</function>'
    matches = re.finditer(pattern1, generated_text, re.DOTALL)
    for match in matches:
        function_name = match.group(1).strip()
        arguments_str = match.group(2).strip()
        try:
            arguments = json.loads(arguments_str)
            tool_calls.append({
                "name": function_name,
                "arguments": arguments
            })
        except json.JSONDecodeError:
            # Skip malformed tool calls
            continue

    # Pattern 2: {"name": "...", "arguments": {...}}
    if not tool_calls:
        # Try to find JSON objects with name and arguments
        pattern2 = r'\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{[^}]*\})[^}]*\}'
        matches = re.finditer(pattern2, generated_text)
        for match in matches:
            function_name = match.group(1)
            arguments_str = match.group(2)
            try:
                arguments = json.loads(arguments_str)
                tool_calls.append({
                    "name": function_name,
                    "arguments": arguments
                })
            except json.JSONDecodeError:
                continue

    # If still no tool calls found, try to extract any JSON-like structure
    if not tool_calls:
        # Look for any JSON object in the text
        try:
            # Find all JSON objects
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            for match in re.finditer(json_pattern, generated_text):
                try:
                    obj = json.loads(match.group(0))
                    if "name" in obj and "arguments" in obj:
                        tool_calls.append(obj)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    return tool_calls
