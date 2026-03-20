"""
pipeline/module_2/response_schema.py - Dynamic response schema factory.

Generates an OpenAI structured-output JSON schema constrained to exactly
the two response options for a given illusion.
"""


def make_schema(response_options: list[str]) -> dict:
    """
    Build a JSON schema that enforces a forced-choice response.

    Args:
        response_options: Two-element list, e.g. ["Top", "Bottom"].

    Returns:
        Schema dict suitable for the ``text.format`` field of the
        Responses API ``responses.create()`` call.
    """
    option_a, option_b = response_options
    name = f"forced_choice_{'_'.join(opt.lower() for opt in response_options)}"

    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "response": {
                    "type": "string",
                    "enum": response_options,
                    "description": (
                        f"Which option looks greater: " f'"{option_a}" or "{option_b}".'
                    ),
                },
            },
            "required": ["image_id", "response"],
            "additionalProperties": False,
        },
    }


def make_chat_completions_schema(response_options: list[str]) -> dict:
    """
    Build a JSON schema in chat.completions format (used by the Batch API).

    The Batch API targets /v1/chat/completions which uses ``response_format``
    with a slightly different wrapper than the Responses API.

    Args:
        response_options: Two-element list, e.g. ["Top", "Bottom"].

    Returns:
        Schema dict suitable for the ``response_format`` field of a
        chat.completions request.
    """
    base = make_schema(response_options)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": base["name"],
            "strict": base["strict"],
            "schema": base["schema"],
        },
    }
