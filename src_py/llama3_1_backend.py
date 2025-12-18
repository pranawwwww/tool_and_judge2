"""
Llama 3.1 backend implementation using vLLM.

This backend implements Llama 3.1's tool calling format as specified in:
https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct#tool-use-with-transformers
"""

import json
import re
from typing import Any, List

from vllm import SamplingParams


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
    return generated_text

    # # Parse the tool calls from the generated text
    # # Llama 3.1 outputs tool calls in a specific format that needs to be extracted
    # tool_calls = parse_llama3_1_tool_calls(generated_text)

    # # Convert to the expected output format
    # response_dicts = []
    # for tool_call in tool_calls:
    #     # Check if the tool call has required fields
    #     if "name" not in tool_call:
    #         continue  # Skip malformed tool calls without a name

    #     # Get arguments, default to empty dict if not present
    #     arguments = tool_call.get("arguments", {})

    #     response_dicts.append({
    #         "type": "function",
    #         "function": {
    #             "name": tool_call["name"],
    #             "arguments": json.dumps(arguments)
    #         }
    #     })

    # response_json_str = json.dumps(response_dicts)
    # return response_json_str


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
            # Validate and normalize each item
            normalized = []
            for item in parsed:
                if isinstance(item, dict) and "name" in item:
                    normalized.append({
                        "name": item["name"],
                        "arguments": item.get("arguments", {})
                    })
            if normalized:
                return normalized
        elif isinstance(parsed, dict) and "name" in parsed:
            return [{
                "name": parsed["name"],
                "arguments": parsed.get("arguments", {})
            }]
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
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({
                "name": function_name,
                "arguments": arguments
            })

    # If still no tool calls found, try to extract any JSON-like structure
    if not tool_calls:
        # Look for any JSON object in the text
        try:
            # Find all JSON objects
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            for match in re.finditer(json_pattern, generated_text):
                try:
                    obj = json.loads(match.group(0))
                    if "name" in obj:
                        tool_calls.append({
                            "name": obj["name"],
                            "arguments": obj.get("arguments", {})
                        })
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    # Ensure all tool calls have the required structure
    normalized_tool_calls = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict) and "name" in tool_call:
            normalized_tool_calls.append({
                "name": tool_call["name"],
                "arguments": tool_call.get("arguments", {})
            })

    return normalized_tool_calls


def collect_perplexity_batch(
    entries: List[dict],
    model: Any,
    tokenizer: Any,
) -> List[dict]:
    """
    Collect raw logits and input_ids for a batch of entries using HuggingFace backend.

    Args:
        entries: List of entries, each containing 'question' and 'answer' fields
        model: HuggingFace model instance
        tokenizer: Tokenizer instance

    Returns:
        List of dicts containing 'logits' and 'input_ids' for each entry
    """
    import torch
    from src_py.utils import language_abbreviation_to_name

    results = []

    for entry in entries:
        question = entry['question']
        answer = entry['answer']
        lang = entry.get('lang', 'en')

        # Map language abbreviation to full name
        language_name = language_abbreviation_to_name(lang)

        # Build language-specific instructions (following qwen3_interface.py format)
        instruction = f"Please answer the question in {language_name} with a concise phrase instead of a complete sentence."

        # Combine question with instruction
        user_content = f"{question}\n\n{instruction}"

        # Build messages for chat template
        messages = [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        # Apply chat template to get the full formatted prompt
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize the formatted prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids[0]

        # Move input to model's device
        input_ids_tensor = input_ids.unsqueeze(0).to(model.device)

        # Get logits from model
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            logits = outputs.logits[0].cpu()  # [seq_len, vocab_size], move to CPU

        # Store results with input_ids and logits
        results.append({
            'logits': logits,
            'input_ids': input_ids.tolist(),
            'answer': answer  # Keep answer for backward search
        })

    return results


async def collect_preference_local_async(
    question: str,
    answer1: str,
    answer2: str,
    engine: Any,
    tokenizer: Any,
) -> tuple[float, float]:
    """
    Collect preference between two answers using Llama 3.1 backend.

    Returns:
        Tuple of (logprob_1, logprob_2) where:
        - logprob_1: log probability of token "1"
        - logprob_2: log probability of token "2"
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial judge. The user is going to provide one question and two answers. "
                'If Answer 1 is better, respond with "1". '
                'If Answer 2 is better, respond with "2". '
                "Even if the answers are identical in correctness, try your best to choose a more favorable one. "
                "IMPORTANT: Language is not a factor in your judgment; focus solely on the content quality.\n"
                'Only respond with "1" or "2", without any explanation.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Answer 1: {answer1}\n"
                f"Answer 2: {answer2}\n"
                "Which answer is better? Respond with '1' for Answer 1 or '2' for Answer 2."
            ),
        },
    ]

    # Convert chat messages to prompt text
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=1.0,      # sampling at temperature 1.0
        max_tokens=4,         # only need one token
        stop=None,
        logprobs=10,          # get top 10 token log probabilities
    )

    # vLLM async generation (returns an async generator)
    request_id = f"llama31_preference_{id(question)}"
    final_output = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        # We only need the final result
        final_output = output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    # Get the first token's logprobs
    if not final_output.outputs[0].logprobs or len(final_output.outputs[0].logprobs) == 0:
        raise RuntimeError("No logprobs returned from vLLM")

    first_token_logprobs = final_output.outputs[0].logprobs[0]

    # Get token IDs for "1" and "2"
    token_1_id = tokenizer.encode("1", add_special_tokens=False)[0]
    token_2_id = tokenizer.encode("2", add_special_tokens=False)[0]

    # Extract log probabilities for tokens "1" and "2"
    logprob_1 = None
    logprob_2 = None

    for token_id, logprob_obj in first_token_logprobs.items():
        if token_id == token_1_id:
            logprob_1 = logprob_obj.logprob
        elif token_id == token_2_id:
            logprob_2 = logprob_obj.logprob

    # Check if both tokens are in the top k
    if logprob_1 is None:
        raise ValueError(f"Token '1' (ID: {token_1_id}) not found in top-k logprobs")
    if logprob_2 is None:
        raise ValueError(f"Token '2' (ID: {token_2_id}) not found in top-k logprobs")

    return logprob_1, logprob_2
