import json
from typing import Any, Dict

from lmnr import evaluate

from index import AnthropicProvider
from index.agent.utils import generate_proper_json

llm = AnthropicProvider(model="claude-3-7-sonnet-20250219", enable_thinking=True, thinking_token_budget=1024)
    
async def run_json_correction(data: Dict[str, Any]):
    """Execute the JSON correction function."""
    malformed_json = data["malformed_json"]
    # We'll need an LLM provider. Let's use GeminiProvider as in the reference.
    # In a real scenario, you might want to configure this or pass it differently.
    
    corrected_json_str = await generate_proper_json(llm=llm, json_str=malformed_json)
    
    # The function returns a string, let's try to parse it to ensure it's valid JSON for the eval
    try:
        return json.loads(corrected_json_str)
    except json.JSONDecodeError:
        # If it's not valid JSON, return the string itself for the evaluator to handle
        return corrected_json_str


async def eval_json_correction(output: Any, target: Dict[str, Any]):
    """Evaluate the JSON correction accuracy."""
    # Assuming target is a Python dict representing the expected JSON
    # And output is also a Python dict (if parsing was successful) or a string
    
    if isinstance(output, str):
        # This means the corrected_json_str was not valid JSON
        # For this simple eval, we can consider this a failure if the target is a dict
        # Or, if the target itself is expected to be a non-JSON string (e.g. an error message)
        # For now, let's assume target is always a valid JSON object.
        try:
            # Attempt to parse the output string here for comparison
            output_dict = json.loads(output)
            exact_match = output_dict == target
        except json.JSONDecodeError:
            exact_match = False # Output was not valid JSON
    else: # Output is already a dict
        exact_match = output == target
    
    return exact_match

test_data = [
    {
        "data": {
            # Trailing comma, single quotes
            "malformed_json": "{'name': 'John Doe', 'age': 30, 'city': 'New York',}",
        },
        "target": {
            "name": "John Doe",
            "age": 30,
            "city": "New York"
        }
    },
    {
        "data": {
            "malformed_json": '''{
                "item": "Book",
                "details": {
                    "title": "The "Great Gatsby"",
                    "author": "F. Scott Fitzgerald"
                },
                "price": 10.99
            }'''
        },
        "target": {
            "item": "Book",
            "details": {
                "title": "The \"Great Gatsby\"",
                "author": "F. Scott Fitzgerald"
            },
            "price": 10.99
        }
    },
    {
        "data": {
            # No closing brace
            "malformed_json": '''{
                "key1": "value1",
                "key2": "value2"
            ''' # Corrected: Removed trailing content that looked like a comment inside string
        },
        "target": {
            "key1": "value1",
            "key2": "value2"
        }
    },
    {
        "data": {
            # JSON with comments (not standard, should be removed by the fixer)
            "malformed_json": '''{
                // This is a comment
                "product_id": 123,
                "status": "active"
            }'''
        },
        "target": {
            "product_id": 123,
            "status": "active"
        }
    },
    # Example of a more complex malformed JSON
    {
        "data": {
            "malformed_json": "{\"name\": \"incomplete, \"value\": [1, 2, \"unfinished_array\"" # Missing closing bracket and quote
        },
        "target": { # Assuming the LLM can make a reasonable guess or fix structure
            "name": "incomplete",
            "value": [1, 2, "unfinished_array"]
        }
    },
    {
        "data": {
            "malformed_json": "{'key with space': 'value', 'another key': true, 'numeric_string': '123.45' }" # Single quotes, boolean
        },
        "target": {
            "key with space": "value",
            "another key": True, # Python bool
            "numeric_string": "123.45"
        }
    }
]

# Run the evaluation
evaluate(
    data=test_data,
    executor=run_json_correction,
    evaluators={"json_correction_accuracy": eval_json_correction},
    concurrency_limit=10,
    group_name="json_correction_eval",
)
