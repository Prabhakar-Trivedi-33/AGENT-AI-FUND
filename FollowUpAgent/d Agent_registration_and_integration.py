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


# sample_usage.py
"""
Example usage of the Follow-Up Agent in the agent execution pipeline.
"""

import json
import logging
from typing import Dict, Any

from .llm_handler import LLMHandler
from .follow_up_agent import FollowUpAgent, LLMBasedFollowUpGenerator
from .agent_graph import register_follow_up_agent, format_final_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_primary_agent_execution() -> Dict[str, Any]:
    """Simulate execution of a primary agent for demonstration purposes.
    
    Returns:
        Simulated agent state after primary agent execution
    """
    # Sample agent state after portfolio agent execution
    return {
        "user_query": "How is my portfolio performing this year?",
        "current_agent_type": "portfolio_agent",
        "user_profile": {
            "risk_level": "moderate",
            "investment_horizon": "long-term",
            "investment_goals": ["retirement", "education"],
            "knowledge_level": "intermediate"
        },
        "portfolio_data": {
            "total_value": 250000,
            "ytd_return": 0.058,  # 5.8%
            "asset_allocation": {
                "stocks": 0.65,   # 65%
                "bonds": 0.25,    # 25% 
                "cash": 0.10      # 10%
            },
            "top_holdings": [
                {"name": "VTSAX", "allocation": 0.30},
                {"name": "VTIAX", "allocation": 0.20},
                {"name": "VBTLX", "allocation": 0.15}
            ]
        },
        "primary_agent_response": {
            "message": "Your portfolio has performed well this year with a 5.8% return, which is slightly above the benchmark for your allocation. Your asset mix of 65% stocks, 25% bonds, and 10% cash aligns well with your moderate risk profile and long-term horizon. Your largest holdings continue to be in total market index funds, providing good diversification.",
            "charts": ["portfolio_performance", "asset_allocation"],
            "status": "success"
        },
        "conversation_history": [
            {
                "user_query": "What are my current investments?",
                "follow_up_questions": [
                    "How have my investments performed year-to-date?",
                    "Should I rebalance my portfolio?"
                ]
            }
        ],
        "timestamp": "2025-05-01T10:30:00"
    }


def main():
    """Main function demonstrating the use of Follow-Up Agent."""
    logger.info("Starting Follow-Up Agent demonstration")
    
    # Initialize LLM handler (using mock for demonstration)
    class MockLLMHandler(LLMHandler):
        def generate_response(self, prompt: str) -> str:
            logger.info(f"Mock LLM received prompt of {len(prompt)} characters")
            return json.dumps([
                "Would you like to see how your portfolio compares to market benchmarks?",
                "Are you interested in rebalancing options to optimize returns?",
                "Would you like to explore tax implications of your current allocation?"
            ])
    
    llm_handler = MockLLMHandler()
    
    # Create Follow-Up Agent
    follow_up_agent = FollowUpAgent.create_with_llm(llm_handler, max_questions=3)
    
    # Simulate primary agent execution
    agent_state = simulate_primary_agent_execution()
    logger.info("Primary agent execution completed")
    
    # Process with Follow-Up Agent
    updated_state = follow_up_agent.process(agent_state)
    logger.info(f"Follow-Up Agent generated {len(updated_state.get('follow_up_questions', []))} questions")
    
    # Format final response
    final_response = format_final_response(updated_state)
    
    # Print result
    logger.info("Final response with follow-up questions:")
    print(json.dumps(final_response, indent=2))
    
    # Example of how this would be integrated in the agent graph
    logger.info("\nExample of agent graph integration:")
    print("""
    # In your main application:
    from agent_graph import AgentGraph, register_follow_up_agent
    
    # Initialize components
    llm_handler = LLMHandler()
    agent_graph = AgentGraph()
    
    # Register primary agents
    agent_graph.register_agent(PortfolioAgent(), "portfolio_agent")
    agent_graph.register_agent(FundAgent(), "fund_agent")
    # ... other primary agents
    
    # Register Follow-Up Agent as post-processor
    register_follow_up_agent(agent_graph, llm_handler)
    
    # Execute agent pipeline
    result = agent_graph.execute("How is my portfolio performing?")
    # Result will include follow-up questions automatically
    """)


if __name__ == "__main__":
    main()
