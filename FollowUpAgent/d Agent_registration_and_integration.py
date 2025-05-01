# agent_graph.py (Example file showing how to register the Follow-Up Agent)
"""
Agent graph configuration for integrating the Follow-Up Agent as a post-processor.
"""

from typing import Dict, Any, List

from .follow_up_agent import FollowUpAgent
from .llm_handler import LLMHandler


def register_follow_up_agent(agent_graph, llm_handler: LLMHandler):
    """Register the Follow-Up Agent in the agent graph as a post-processor.
    
    Args:
        agent_graph: The agent graph system
        llm_handler: LLM handler for making inference calls
    """
    # Create the Follow-Up Agent
    follow_up_agent = FollowUpAgent.create_with_llm(
        llm_handler=llm_handler,
        max_questions=3  # Configure as needed
    )
    
    # Register agent as post-processor for all primary agents
    # This is an example - adjust to match your agent graph system
    
    # Option 1: Register as a global post-processor
    agent_graph.register_post_processor(
        agent=follow_up_agent,
        name="follow_up_agent"
    )
    
    # Option 2: Register as post-processor for specific agents
    primary_agents = [
        "portfolio_agent",
        "fund_agent",
        "tax_agent",
        "recommendation_agent"
    ]
    
    for agent_name in primary_agents:
        agent_graph.register_edge(
            source=agent_name,
            target="follow_up_agent",
            edge_type="post_process"
        )
    
    # Ensure the follow-up agent response is included in the final response
    agent_graph.configure_response_handler(include_follow_up_questions=True)


# Example of a response handler/formatter that incorporates follow-up questions
def format_final_response(agent_state: Dict[str, Any]) -> Dict[str, Any]:
    """Format the final response to include follow-up questions.
    
    Args:
        agent_state: The agent state after processing
        
    Returns:
        Formatted response with follow-up questions
    """
    # Extract primary response
    if "primary_agent_response" not in agent_state:
        return {"error": "No primary agent response found"}
    
    primary_response = agent_state["primary_agent_response"]
    
    # Extract follow-up questions if available
    follow_up_questions = agent_state.get("follow_up_questions", [])
    
    # Construct final response
    final_response = {
        # Include all fields from primary response
        **primary_response,
        
        # Add follow-up questions field
        "follow_up_
