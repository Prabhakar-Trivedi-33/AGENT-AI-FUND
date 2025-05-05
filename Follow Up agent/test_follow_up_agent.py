import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from services.agents.follow_up_agent.follow_up_agent import FollowUpAgent
from services.agents.base.state import AgentState
from services.agents.dtos.agent_system_dtos import AgentSystemRequest
from services.llm_service.llm_chat_service import LLMChatService
from services.llm_service.model_enum import ModelProviderEnum

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockLLMChatService:
    """Comprehensive mock LLM Chat Service for testing"""
    def __init__(self, model=None):
        self.model = model
        self.call_count = 0
    
    def generate_response(self, prompt: str) -> str:
        """
        Advanced mock response generation with logging and tracking
        
        :param prompt: Input prompt
        :return: Mocked response
        """
        self.call_count += 1
        logger.info(f"Mock LLM Service called with prompt: {prompt}")
        
        # Predefined responses based on input
        if "intent" in prompt.lower():
            return "Information Seeking"
        elif "follow-up questions" in prompt.lower():
            return "Could you elaborate on your investment goals?\nWhat is your risk tolerance?"
        elif "classification" in prompt.lower():
            return "GENERAL_INFORMATION"
        
        return "Mocked generic response"

@pytest.fixture(scope="function")
def mock_llm_service():
    """Fixture to create a mock LLM service"""
    return MockLLMChatService()

@pytest.fixture(scope="function")
def follow_up_agent(mock_llm_service):
    """
    Fixture to create a FollowUpAgent instance for testing
    
    :param mock_llm_service: Mocked LLM service
    :return: FollowUpAgent instance
    """
    try:
        with patch('services.agents.follow_up_agent.follow_up_agent.LLMChatService', return_value=mock_llm_service):
            agent = FollowUpAgent()
            agent.llm_service = mock_llm_service
            return agent
    except Exception as e:
        logger.error(f"Error creating follow-up agent: {e}")
        raise

@pytest.fixture(scope="function")
def mock_agent_state():
    """
    Create a mock AgentState for testing
    
    :return: AgentState instance
    """
    return AgentState(
        agent_request=AgentSystemRequest(
            query="Tell me about mutual funds",
            messages=[
                Mock(content="Initial query about mutual funds")
            ]
        ),
        agent_response={},
        completed_agents=[],
        next_agent=None
    )

def test_follow_up_agent_initialization(follow_up_agent):
    """
    Test the initialization of FollowUpAgent
    
    :param follow_up_agent: Fixture providing FollowUpAgent instance
    """
    assert follow_up_agent is not None, "Follow-up agent should be initialized"
    assert follow_up_agent.agent_name == "FollowUp", "Agent name should be 'FollowUp'"
    assert hasattr(follow_up_agent, 'llm_service'), "Agent should have LLM service"

def test_context_extraction(follow_up_agent, mock_agent_state):
    """
    Comprehensive test for context extraction methods
    
    :param follow_up_agent: Fixture providing FollowUpAgent instance
    :param mock_agent_state: Fixture providing mock AgentState
    """
    # Test intent extraction
    intent = follow_up_agent._extract_intent(mock_agent_state)
    assert intent is not None, "Intent should be extracted"
    assert isinstance(intent, str), "Intent should be a string"

    # Test additional context extraction
    mock_agent_state.fund_data = {"test": "data"}
    mock_agent_state.portfolio_data = {"portfolio": "details"}
    
    additional_context = follow_up_agent._extract_additional_context(mock_agent_state)
    assert "fund_data" in additional_context, "Fund data should be in additional context"
    assert "portfolio_data" in additional_context, "Portfolio data should be in additional context"

def test_follow_up_question_generation(follow_up_agent, mock_agent_state):
    """
    Test follow-up question generation
    
    :param follow_up_agent: Fixture providing FollowUpAgent instance
    :param mock_agent_state: Fixture providing mock AgentState
    """
    context = follow_up_agent._prepare_follow_up_context(mock_agent_state)
    
    # Generate follow-up questions
    follow_up_result = follow_up_agent._generate_follow_up_questions(context)
    
    assert "follow_up_questions" in follow_up_result, "Follow-up result should contain questions"
    assert "reasoning" in follow_up_result, "Follow-up result should have reasoning"
    assert "confidence_score" in follow_up_result, "Follow-up result should have confidence score"
    
    questions = follow_up_result["follow_up_questions"]
    assert isinstance(questions, list), "Follow-up questions should be a list"
    assert all(isinstance(q, str) and q.endswith('?') for q in questions), "All questions should be strings ending with '?'"

def test_question_validation(follow_up_agent):
    """
    Test follow-up question validation
    
    :param follow_up_agent: Fixture providing FollowUpAgent instance
    """
    # Valid questions
    valid_questions = [
        "Could you provide more details about your investment goals?",
        "What specific mutual funds are you interested in?"
    ]
    assert follow_up_agent._validate_follow_up_questions(valid_questions) is True, "Valid questions should pass validation"
    
    # Invalid questions
    invalid_questions = [
        "Short",  # Too short
        "No question mark",
        "A" * 500  # Too long
    ]
    assert follow_up_agent._validate_follow_up_questions(invalid_questions) is False, "Invalid questions should fail validation"

def test_execute_agent_full_flow(follow_up_agent, mock_agent_state):
    """
    Test the full execution flow of the follow-up agent
    
    :param follow_up_agent: Fixture providing FollowUpAgent instance
    :param mock_agent_state: Fixture providing mock AgentState
    """
    updated_state = follow_up_agent.execute_agent(mock_agent_state)
    
    # Verify state updates
    assert hasattr(updated_state, 'follow_up_questions'), "Updated state should have follow-up questions"
    assert hasattr(updated_state, 'follow_up_reasoning'), "Updated state should have follow-up reasoning"
    assert hasattr(updated_state, 'follow_up_confidence'), "Updated state should have follow-up confidence"
    
    # Validate follow-up questions
    assert len(updated_state.follow_up_questions) > 0, "At least one follow-up question should be generated"
    assert all(isinstance(q, str) and q.endswith('?') for q in updated_state.follow_up_questions), "Questions should be valid"

def test_error_handling(follow_up_agent, mock_agent_state):
    """
    Test error handling in the follow-up agent
    
    :param follow_up_agent: Fixture providing FollowUpAgent instance
    :param mock_agent_state: Fixture providing mock AgentState
    """
    # Simulate an error during follow-up question generation
    with patch.object(follow_up_agent, '_generate_follow_up_questions', side_effect=Exception("Test error")):
        updated_state = follow_up_agent.execute_agent(mock_agent_state)
        
        # Verify error handling
        assert hasattr(updated_state, 'error'), "Updated state should have error attribute"
        assert updated_state.error is not None, "Error should be set"

@pytest.mark.parametrize("scenarios", [
    {
        "query": "Tell me about mutual funds",
        "expected_intents": ["Information Seeking", "Recommendation"]
    },
    {
        "query": "Compare ICICI and HDFC mutual funds",
        "expected_intents": ["Comparison"]
    },
    {
        "query": "What should I invest in?",
        "expected_intents": ["Recommendation", "Problem Solving"]
    }
])
def test_comprehensive_scenarios(follow_up_agent, scenarios):
    """
    Comprehensive test covering multiple scenarios
    
    :param follow_up_agent: Fixture providing FollowUpAgent instance
    :param scenarios: Parameterized test scenarios
    """
    mock_state = AgentState(
        agent_request=AgentSystemRequest(
            query=scenarios["query"],
            messages=[Mock(content=scenarios["query"])]
        ),
        agent_response={},
        completed_agents=[],
        next_agent=None
    )
    
    updated_state = follow_up_agent.execute_agent(mock_state)
    
    # Verify follow-up generation
    assert hasattr(updated_state, 'follow_up_questions'), "Updated state should have follow-up questions"
    assert len(updated_state.follow_up_questions) > 0, "At least one follow-up question should be generated"
    assert all(isinstance(q, str) and q.endswith('?') for q in updated_state.follow_up_questions), "Questions should be valid"

def test_performance_with_large_context(follow_up_agent):
    """
    Test agent performance with a large context
    
    :param follow_up_agent: Fixture providing FollowUpAgent instance
    """
    # Create a mock state with a very large conversation history
    large_state = AgentState(
        agent_request=AgentSystemRequest(
            query="Detailed query about investment strategy",
            messages=[Mock(content=f"Message {i}") for i in range(100)]  # 100 messages
        ),
        agent_response={},
        completed_agents=[],
        next_agent=None
    )
    
    # Add some additional data to simulate a complex state
    large_state.fund_data = {"complex": "fund_data" * 50}
    large_state.portfolio_data = {"large": "portfolio" * 50}
    
    # Execute the agent
    updated_state = follow_up_agent.execute_agent(large_state)
    
    # Verify basic expectations
    assert hasattr(updated_state, 'follow_up_questions'), "Large context state should have follow-up questions"
    assert len(updated_state.follow_up_questions) > 0, "At least one follow-up question should be generated" 
