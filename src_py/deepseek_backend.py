"""
DeepSeek backend for tool calling.

This backend supports DeepSeek's native function calling API which uses
OpenAI-compatible interface with structured tool_calls in responses.

Reference: https://api-docs.deepseek.com/guides/function_calling
"""

import json
from typing import Any, List, Dict


async def generate_tool_call_async(
    model_name: str,
    client: Any,
    question: str,
    tools: List[Dict],
    prompt_passing_in_english: bool
) -> str:
    """
    Generate tool calls using DeepSeek API.

    Args:
        model_name: DeepSeek model name (e.g., "deepseek-chat")
        client: OpenAI-compatible async client
        question: User query
        tools: List of tool definitions in OpenAI format
        prompt_passing_in_english: Whether to request English parameter passing

    Returns:
        JSON string containing the model's response with tool calls
    """
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
            "Do NOT attempt to answer the question directly without using the available functions. "
            f"{'IMPORTANT: Pass all parameter values in English' if prompt_passing_in_english else ''}"
        )
    }

    input_messages = [
        system_message,
        {"role": "user", "content": question}
    ]

    response = await client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        tools=tools,
        temperature=0.0,
    )

    # Convert response to JSON format similar to gpt5_backend
    response_dict = response.choices[0].message.model_dump(exclude_none=True)
    return json.dumps(response_dict)


async def translate_tool_question_async(model_name: str, client: Any, question: str) -> str:
    """
    Translate a user question to English using DeepSeek.

    Args:
        model_name: DeepSeek model name
        client: OpenAI-compatible async client
        question: Question to translate

    Returns:
        Translated question as string
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

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()


async def translate_tool_answer_async(model_name: str, client: Any, parameter_value: str) -> str:
    """
    Translate a parameter value to English using DeepSeek.

    Args:
        model_name: DeepSeek model name
        client: OpenAI-compatible async client
        parameter_value: Parameter value to translate

    Returns:
        Translated parameter value as string
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

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()
