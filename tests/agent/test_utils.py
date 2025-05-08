# Test cases for agent utility functions 

import pytest

from index.agent.models import (  # Assuming ActionModel is part of AgentLLMOutput
    ActionModel,
    AgentLLMOutput,
)
from index.agent.utils import generate_proper_json, validate_json
from index.llm.llm import (  # Assuming LLMResponse is the type returned by llm.call
    BaseLLMProvider,
    LLMResponse,
    Message,
)


# Mock LLM Provider
class MockLLMProvider(BaseLLMProvider):
    def __init__(self, responses=None, call_should_fail=False, exception_to_raise=None):
        self.responses = responses if responses is not None else []
        self.call_history = []
        self.call_should_fail = call_should_fail
        self.exception_to_raise = exception_to_raise if exception_to_raise else Exception("LLM call failed")

    async def call(self, messages: list[Message]) -> LLMResponse:
        self.call_history.append(messages)
        if self.call_should_fail:
            raise self.exception_to_raise
        if self.responses:
            response_content = self.responses.pop(0)
            # Simulate LLMResponse structure; adjust if it's different
            return LLMResponse(content=response_content, thinking=None, raw_response=None, cost=None, usage={"prompt_tokens": 10, "completion_tokens": 10})
        return LLMResponse(content="", thinking=None, raw_response=None, cost=None, usage={"prompt_tokens": 0, "completion_tokens": 0}) # Default empty response

    def get_token_limit(self) -> int:
        return 4096 # Dummy value

    def count_tokens(self, text: str) -> int:
        return len(text.split()) # Dummy value

# --- Tests for validate_json ---

@pytest.mark.asyncio
async def test_validate_json_valid_with_output_tags():
    raw_response = "<output_123>{\"action\": {\"name\": \"click\", \"params\": {\"selector\": \".btn\"}}, \"thought\": \"Thinking...\", \"summary\": \"Clicked button\"}</output_123>"
    mock_llm = MockLLMProvider()
    
    expected_action = ActionModel(name="click", params={"selector": ".btn"})
    expected_output = AgentLLMOutput(action=expected_action, thought="Thinking...", summary="Clicked button")
    
    result = await validate_json(raw_response, mock_llm)
    
    assert result.action == expected_action
    assert result.thought == expected_output.thought
    assert result.summary == expected_output.summary
    assert len(mock_llm.call_history) == 0 # LLM should not be called

@pytest.mark.asyncio
async def test_validate_json_valid_with_json_markdown():
    raw_response = "```json\n{\"action\": {\"name\": \"type\", \"params\": {\"text\": \"hello\"}}, \"thought\": \"Typing...\", \"summary\": \"Typed hello\"}\n```"
    mock_llm = MockLLMProvider()

    expected_action = ActionModel(name="type", params={"text": "hello"})
    expected_output = AgentLLMOutput(action=expected_action, thought="Typing...", summary="Typed hello")

    result = await validate_json(raw_response, mock_llm)

    assert result.action == expected_action
    assert result.thought == expected_output.thought
    assert result.summary == expected_output.summary
    assert len(mock_llm.call_history) == 0

@pytest.mark.asyncio
async def test_validate_json_valid_plain_json_no_tags_no_markdown():
    raw_response = "{\"action\": {\"name\": \"scroll\", \"params\": {\"direction\": \"down\"}}, \"thought\": \"Scrolling...\", \"summary\": \"Scrolled down\"}"
    mock_llm = MockLLMProvider()

    expected_action = ActionModel(name="scroll", params={"direction": "down"})
    expected_output = AgentLLMOutput(action=expected_action, thought="Scrolling...", summary="Scrolled down")
    
    result = await validate_json(raw_response, mock_llm)

    assert result.action == expected_action
    assert result.thought == expected_output.thought
    assert result.summary == expected_output.summary
    assert len(mock_llm.call_history) == 0

@pytest.mark.asyncio
async def test_validate_json_needs_cleaning_escaped_chars():
    # Contains \\n which should be cleaned to \n by the first cleaning pass
    # Changed input to use standard JSON escape \n instead of \\\\n
    raw_response = "<output>{\"action\": {\"name\": \"navigate\", \"params\": {\"url\": \"test.com\"}}, \"thought\": \"Navigating...\\nNext line.\", \"summary\": \"Navigated\"}</output>"
    mock_llm = MockLLMProvider()

    expected_action = ActionModel(name="navigate", params={"url": "test.com"})
    # Expected output still has a real newline
    expected_output = AgentLLMOutput(action=expected_action, thought="Navigating...\nNext line.", summary="Navigated")

    result = await validate_json(raw_response, mock_llm)

    assert result.action == expected_action
    assert result.thought == expected_output.thought # Direct comparison
    assert result.summary == expected_output.summary
    assert len(mock_llm.call_history) == 0

@pytest.mark.asyncio
async def test_validate_json_needs_cleaning_control_chars():
    # Contains a control character (bell \x07) that should be removed
    raw_response = "<output>{\"action\": {\"name\": \"wait\", \"params\": {}}, \"thought\": \"Waiting...\x07\", \"summary\": \"Waited\"}</output>"
    mock_llm = MockLLMProvider()
    
    expected_action = ActionModel(name="wait", params={})
    expected_output = AgentLLMOutput(action=expected_action, thought="Waiting...", summary="Waited")

    result = await validate_json(raw_response, mock_llm)
    
    assert result.action.name == expected_action.name
    assert result.action.params == expected_action.params
    assert result.thought == expected_output.thought
    assert result.summary == expected_output.summary
    assert len(mock_llm.call_history) == 0

# --- Tests for generate_proper_json (can be simple, as it's a direct LLM call) ---
@pytest.mark.asyncio
async def test_generate_proper_json_calls_llm_and_strips():
    mock_llm = MockLLMProvider(responses=["  ```json\n{\"key\": \"fixed_value\"}```  "])
    malformed_json = "{key: 'broken_value'"
    
    result = await generate_proper_json(mock_llm, malformed_json)
    
    assert result == "{\"key\": \"fixed_value\"}"
    assert len(mock_llm.call_history) == 1
    # Check prompt content
    assert "Problematic JSON string:" in mock_llm.call_history[0][0].content[0].text
    assert malformed_json in mock_llm.call_history[0][0].content[0].text

# More tests for validate_json involving LLM fixes and failures will follow

@pytest.mark.asyncio
async def test_validate_json_llm_fix_succeeds_on_first_llm_call():
    malformed_raw_response = "<output>{\"action\": {\"name\": \"bad\", \"params\": {\"selector\": \".err\"}}, \"thought\": \"Oops\"summary\": \"Bad JSON\"}</output>"
    corrected_json_str = "{\"action\": {\"name\": \"fixed\", \"params\": {\"detail\": \"good\"}}, \"thought\": \"Fixed!\", \"summary\": \"JSON is now good\"}"
    
    mock_llm = MockLLMProvider(responses=[corrected_json_str])
    
    expected_action = ActionModel(name="fixed", params={"detail": "good"})
    expected_output = AgentLLMOutput(action=expected_action, thought="Fixed!", summary="JSON is now good")

    result = await validate_json(malformed_raw_response, mock_llm)

    assert result.action == expected_action
    assert result.thought == expected_output.thought
    assert result.summary == expected_output.summary
    assert len(mock_llm.call_history) == 1 # LLM called once to fix
    assert "Problematic JSON string:" in mock_llm.call_history[0][0].content[0].text
    # The string passed to LLM should be the extracted content from output tags
    assert "{\"action\": {\"name\": \"bad\", \"params\": {\"selector\": \".err\"}}, \"thought\": \"Oops\"summary\": \"Bad JSON\"}" in mock_llm.call_history[0][0].content[0].text

@pytest.mark.asyncio
async def test_validate_json_llm_fix_succeeds_after_one_failed_llm_fix_attempt():
    malformed_raw_response = "<output>this is very broken</output>"
    still_malformed_json_from_llm1 = "{still: \"broken\""
    corrected_json_str_from_llm2 = "{\"action\": {\"name\": \"finally_fixed\", \"params\": {}}, \"thought\": \"Phew!\", \"summary\": \"Fixed on second try\"}"
    
    mock_llm = MockLLMProvider(responses=[still_malformed_json_from_llm1, corrected_json_str_from_llm2])
    
    expected_action = ActionModel(name="finally_fixed", params={})
    expected_output = AgentLLMOutput(action=expected_action, thought="Phew!", summary="Fixed on second try")

    result = await validate_json(malformed_raw_response, mock_llm, max_retries=3)

    assert result.action.name == expected_action.name
    assert result.action.params == expected_action.params
    assert result.thought == expected_output.thought
    assert result.summary == expected_output.summary
    assert len(mock_llm.call_history) == 2 # LLM called twice
    # Check what was sent to LLM on first call
    assert "this is very broken" in mock_llm.call_history[0][0].content[0].text 
    # Check what was sent to LLM on second call
    assert still_malformed_json_from_llm1 in mock_llm.call_history[1][0].content[0].text

@pytest.mark.asyncio
async def test_validate_json_fails_after_max_retries_with_llm():
    malformed_raw_response = "<output>totally unfixable {</output>"
    bad_fix1 = "{attempt1: 'bad'"
    bad_fix2 = "{attempt2: 'still bad'"
    bad_fix3 = "{attempt3: 'nope'" # Assuming max_retries is 3 by default in validate_json
    
    mock_llm = MockLLMProvider(responses=[bad_fix1, bad_fix2, bad_fix3])
    
    with pytest.raises(ValueError) as excinfo:
        await validate_json(malformed_raw_response, mock_llm, max_retries=3)
    
    assert "Could not parse or validate response after 3 attempts" in str(excinfo.value)
    assert len(mock_llm.call_history) == 2 # Corrected from 3 to 2
    # The final problematic string in the error message should be the last one LLM produced
    assert f"Final problematic JSON string after all attempts: '{bad_fix2[:500]}" in str(excinfo.value) # LLM is called twice, so bad_fix2 is the last output from LLM

@pytest.mark.asyncio
async def test_validate_json_empty_string_after_extraction():
    # Scenario: <output></output> or <output>   </output>
    raw_response = "<output>  </output>"
    mock_llm = MockLLMProvider() # Returns empty string by default

    with pytest.raises(ValueError) as excinfo:
        await validate_json(raw_response, mock_llm)

    assert "Could not parse or validate response" in str(excinfo.value)
    assert "Final problematic JSON string after all attempts: '...'" in str(excinfo.value)
    # LLM is called max_retries - 1 = 2 times in this path
    assert len(mock_llm.call_history) == 2

@pytest.mark.asyncio
async def test_validate_json_llm_call_itself_fails():
    malformed_raw_response = "<output>broken { </output>"
    mock_llm = MockLLMProvider(call_should_fail=True, exception_to_raise=RuntimeError("LLM service down"))

    with pytest.raises(ValueError) as excinfo:
        await validate_json(malformed_raw_response, mock_llm, max_retries=3)

    assert "Could not parse or validate response after 3 attempts" in str(excinfo.value)
    assert len(mock_llm.call_history) == 2 # Ensure LLM call count is 2
    # Check that the error message ENDS with the expected final string part
    expected_ending = "Final problematic JSON string after all attempts: 'broken {...'"
    assert str(excinfo.value).endswith(expected_ending)

@pytest.mark.asyncio
async def test_validate_json_llm_fix_unescaped_quotes():
    # Input has unescaped double quotes inside string values
    malformed_core = '''{
    "action": {
    "name": "click_element",
    "params": {
        "index": 24,
        "wait_after_click": true
    }
    },
    "thought": "The available options for batches are "ik12" (index 24).",
    "summary": "Trying to click on "ik12" which could be X25."
}
    '''
    malformed_raw_response = f"<output_7>{malformed_core.strip()}</output_7>"

    # Expected corrected JSON from LLM (with escaped quotes)
    corrected_json_string = """
{
    "action": {
        "name": "click_element",
        "params": {
            "index": 24,
            "wait_after_click": true
        }
    },
    "thought": "The available options for batches are \\\"ik12\\\" (index 24).",
    "summary": "Trying to click on \\\"ik12\\\" which could be X25."
}
"""

    # Mock LLM returns the corrected version on the first call
    mock_llm = MockLLMProvider(responses=[corrected_json_string.strip()])

    # Expected Python object representation
    expected_action = ActionModel(
        name="click_element",
        params={"index": 24, "wait_after_click": True}
    )
    expected_thought = 'The available options for batches are "ik12" (index 24).'
    expected_summary = 'Trying to click on "ik12" which could be X25.'

    # Run the validation
    result = await validate_json(malformed_raw_response, mock_llm)

    # Assertions
    assert result.action == expected_action
    assert result.thought == expected_thought
    assert result.summary == expected_summary
    assert len(mock_llm.call_history) == 1 # LLM should be called exactly once
    # Check that the LLM was called with the initially extracted (malformed) string
    assert malformed_core.strip() in mock_llm.call_history[0][0].content[0].text