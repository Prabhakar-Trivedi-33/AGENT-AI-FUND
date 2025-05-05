# FollowUpAgent Technical Documentation

## Overview

The `FollowUpAgent` is a specialized component within the Financial Advisory Agent System designed to identify information gaps in user queries and generate appropriate follow-up questions. This agent is crucial for improving the quality of responses by ensuring all necessary information is gathered before providing financial advice or fund information.

## Key Features

- Contextual analysis of user queries to identify missing information
- Generation of targeted follow-up questions based on conversational context
- Integration with existing conversation history and agent state
- Structured parsing of follow-up responses for downstream processing

## Implementation Details

### Class Structure

```python
class FollowUpAgent(BaseAgent):
    def __init__(self)
    def execute_agent(self, state: AgentState) -> AgentState
    def _format_conversation_history(self, state: AgentState) -> List[Dict]
    def _extract_context(self, state: AgentState) -> Dict
    def _generate_follow_up_questions(self, query: str, conversation_history: List[Dict], current_context: Dict) -> Dict
    def _parse_follow_up_questions(self, response_content: str) -> Dict
    def _identify_missing_information(self, response_content: str) -> List[str]
    def _identify_clarification_needed(self, response_content: str) -> List[str]
```

### Dependencies

- `BaseAgent`: Parent class providing common agent functionality
- `AgentState`: Data structure for maintaining conversation context
- `LLMChatService`: Interface for language model integration
- `ModelProviderEnum`: Enumeration of supported LLM providers

### Prompt Template

The agent uses a specialized prompt template to guide the language model in generating appropriate follow-up questions:

```python
self.prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Follow-Up agent designed to gather additional information from users.
    
    When users provide incomplete queries or requests that need clarification, you should:
    1. Identify what information is missing or unclear
    2. Ask specific questions to obtain the necessary details
    3. Be conversational and helpful in your approach
    4. Focus only on obtaining information relevant to financial/investment inquiries
    
    Examples of follow-up scenarios:
    - User mentions a fund without specifying which one
    - User asks for performance without specifying timeframe
    - User requests comparisons without clarifying the benchmark
    - User asks about portfolio without providing context
    
    Format your follow-up questions clearly and concisely."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])
```

## Method Descriptions

### `execute_agent(state: AgentState) -> AgentState`

The main entry point for agent execution. Processes the current state and generates follow-up questions based on identified gaps.

**Parameters:**
- `state`: Current agent state containing user query and context

**Returns:**
- Updated state with follow-up questions and additional context

**Example:**
```python
follow_up_agent = FollowUpAgent()
updated_state = follow_up_agent.execute_agent(current_state)
```

### `_format_conversation_history(state: AgentState) -> List[Dict]`

Extracts and formats the conversation history from the agent state for use in follow-up generation.

**Parameters:**
- `state`: Current agent state

**Returns:**
- List of formatted conversation messages

### `_extract_context(state: AgentState) -> Dict`

Extracts relevant context from the agent state for improved follow-up generation.

**Parameters:**
- `state`: Current agent state

**Returns:**
- Dictionary containing context information

### `_generate_follow_up_questions(query: str, conversation_history: List[Dict], current_context: Dict) -> Dict`

Core method that generates follow-up questions based on user query and available context.

**Parameters:**
- `query`: User's last query
- `conversation_history`: Previous conversation exchanges
- `current_context`: Relevant context information

**Returns:**
- Dictionary containing follow-up data and response

### `_parse_follow_up_questions(response_content: str) -> Dict`

Parses LLM response to extract structured follow-up question data.

**Parameters:**
- `response_content`: Raw response from the LLM

**Returns:**
- Dictionary containing parsed follow-up information

### `_identify_missing_information(response_content: str) -> List[str]`

Identifies missing information categories from the LLM response.

**Parameters:**
- `response_content`: Raw response from the LLM

**Returns:**
- List of missing information categories

### `_identify_clarification_needed(response_content: str) -> List[str]`

Identifies areas that need clarification from the LLM response.

**Parameters:**
- `response_content`: Raw response from the LLM

**Returns:**
- List of areas needing clarification

## Usage Examples

### Basic Usage

```python
from services.agents.follow_up_agent import FollowUpAgent
from services.agents.base.state import AgentState
from services.agents.base.request import AgentRequest

# Initialize agent
follow_up_agent = FollowUpAgent()

# Create state with user query
state = AgentState(agent_request=AgentRequest(query="How is the fund performing?"))

# Execute follow-up agent
updated_state = follow_up_agent.execute_agent(state)

# Access follow-up questions
follow_up_questions = updated_state.follow_up_data.get("follow_up_questions", [])
follow_up_response = updated_state.agent_response
```

### Integration with Other Agents

```python
# Process initial query with fund information agent
fund_info_agent = FundInformationAgent()
state = fund_info_agent.execute_agent(initial_state)

# Check if follow-up is needed
if not state.fund_data or len(state.fund_data.get("data", {})) < 2:
    # Information is insufficient, use follow-up agent
    follow_up_agent = FollowUpAgent()
    state = follow_up_agent.execute_agent(state)
    
    # Present follow-up questions to user
    # ...

    # Update state with user response
    # ...

    # Re-process with fund information agent
    state = fund_info_agent.execute_agent(state)
```

## Reusable Components

Several components of the `FollowUpAgent` can be reused in other parts of the system:

1. **Conversation History Formatter**: The `_format_conversation_history` method can be extracted as a utility function for any agent that needs to process conversation history.

2. **Context Extractor**: The `_extract_context` logic can be generalized for broader use throughout the agent system.

3. **Response Parsers**: The parsing methods could be moved to a shared utility class for use by multiple agents.

## Error Handling

The agent includes logging at key points to facilitate debugging:

```python
logger.info(f"Follow-up questions generated: {state.agent_response}")
logger.info(f"Generated follow-up questions: {follow_up_questions}")
```

## Future Enhancements

1. **Machine Learning Classification**: Implement ML-based classification of missing information types
2. **Response Templates**: Create templated follow-up responses for common information gaps
3. **Multi-turn Follow-up**: Enhance to support complex multi-turn clarification dialogs
4. **Domain-Specific Enhancements**: Add specialized follow-up logic for different financial domains
5. **Sentiment Analysis**: Incorporate user sentiment analysis to adjust follow-up tone

## Testing

Example test cases for the `FollowUpAgent`:

```python
def test_follow_up_agent_missing_fund_name():
    # Test follow-up when fund name is missing
    state = AgentState(agent_request=AgentRequest(query="How is the fund performing?"))
    agent = FollowUpAgent()
    result = agent.execute_agent(state)
    assert "fund" in result.agent_response.lower()
    assert len(result.follow_up_data.get("missing_information", [])) > 0

def test_follow_up_agent_missing_timeframe():
    # Test follow-up when timeframe is missing
    state = AgentState(agent_request=AgentRequest(query="How is ICICI Prudential Balanced Advantage Fund performing?"))
    agent = FollowUpAgent()
    result = agent.execute_agent(state)
    assert "timeframe" in " ".join(result.follow_up_data.get("missing_information", []))
```
