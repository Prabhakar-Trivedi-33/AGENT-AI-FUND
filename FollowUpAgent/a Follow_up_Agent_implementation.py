# follow_up_agent.py 
"""
Follow-Up Agent module that generates contextual follow-up questions based on 
primary agent responses to enhance user engagement.

This agent acts as a post-processor in the agent execution pipeline and returns 
relevant follow-up questions along with the original agent response.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

from .llm_handler import LLMHandler  # Reusing existing LLM interface
from .agent_base import AgentBase
from .follow_up_agent_prompt import (
    generate_follow_up_prompt,
    parse_follow_up_response
)

logger = logging.getLogger(__name__)


class FollowUpGenerator(ABC):
    """Abstract class defining the interface for follow-up question generation strategies."""
    
    @abstractmethod
    def generate_questions(self, agent_state: Dict[str, Any], 
                          original_response: Dict[str, Any],
                          max_questions: int = 3) -> List[str]:
        """Generate follow-up questions based on agent state and original response."""
        pass


class LLMBasedFollowUpGenerator(FollowUpGenerator):
    """Implementation of follow-up question generation using LLM."""
    
    def __init__(self, llm_handler: LLMHandler):
        """Initialize with LLM handler for making inference calls.
        
        Args:
            llm_handler: Handler for LLM API calls
        """
        self.llm_handler = llm_handler
    
    def generate_questions(self, agent_state: Dict[str, Any], 
                          original_response: Dict[str, Any],
                          max_questions: int = 3) -> List[str]:
        """Generate follow-up questions using LLM based on context.
        
        Args:
            agent_state: The current agent state with user context
            original_response: The response from primary agent
            max_questions: Maximum number of follow-up questions to generate
            
        Returns:
            List of relevant follow-up questions
        """
        # Extract context from agent state
        user_query = agent_state.get("user_query", "")
        agent_type = agent_state.get("current_agent_type", "")
        user_context = self._extract_user_context(agent_state)
        conversation_history = agent_state.get("conversation_history", [])
        
        # Generate prompt for LLM
        prompt = generate_follow_up_prompt(
            user_query=user_query,
            agent_response=original_response,
            agent_type=agent_type,
            user_context=user_context,
            conversation_history=conversation_history,
            max_questions=max_questions
        )
        
        # Call LLM for generating questions
        llm_response = self.llm_handler.generate_response(prompt)
        
        # Parse and validate LLM response
        follow_up_questions = parse_follow_up_response(llm_response)
        
        # Filter for uniqueness against history
        filtered_questions = self._filter_duplicate_questions(
            follow_up_questions, 
            conversation_history
        )
        
        return filtered_questions[:max_questions]
    
    def _extract_user_context(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant user context from agent state.
        
        Args:
            agent_state: The current agent state
            
        Returns:
            Dictionary containing user profile information
        """
        context = {}
        
        # Extract user profile if available
        if "user_profile" in agent_state:
            profile = agent_state["user_profile"]
            context["risk_level"] = profile.get("risk_level", "")
            context["investment_horizon"] = profile.get("investment_horizon", "")
            context["investment_goals"] = profile.get("investment_goals", [])
            context["knowledge_level"] = profile.get("knowledge_level", "")
        
        # Extract portfolio information if available
        if "portfolio_data" in agent_state:
            portfolio = agent_state["portfolio_data"]
            context["portfolio_value"] = portfolio.get("total_value", 0)
            context["asset_allocation"] = portfolio.get("asset_allocation", {})
        
        return context
    
    def _filter_duplicate_questions(self, 
                                   new_questions: List[str], 
                                   conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Filter out questions that have been asked before.
        
        Args:
            new_questions: List of newly generated questions
            conversation_history: Previous conversation history
            
        Returns:
            Filtered list of non-duplicate questions
        """
        # Extract previous follow-up questions from history
        previous_questions = []
        for entry in conversation_history:
            if "follow_up_questions" in entry:
                previous_questions.extend(entry["follow_up_questions"])
        
        # Filter out duplicates (case-insensitive comparison)
        filtered_questions = []
        for question in new_questions:
            if not any(q.lower() == question.lower() for q in previous_questions):
                filtered_questions.append(question)
                
        return filtered_questions


class KeywordBasedFollowUpGenerator(FollowUpGenerator):
    """Mock implementation for testing without LLM dependency."""
    
    def __init__(self, keyword_mapping: Optional[Dict[str, List[str]]] = None):
        """Initialize with keyword mapping for question templates.
        
        Args:
            keyword_mapping: Dictionary mapping keywords to follow-up questions
        """
        self.keyword_mapping = keyword_mapping or {
            "portfolio": [
                "Would you like to see your portfolio performance over time?",
                "Are you interested in rebalancing your portfolio?",
                "Would you like to learn more about diversification strategies?"
            ],
            "fund": [
                "Would you like to compare this fund with similar ones?",
                "Are you interested in the fund's historical performance?",
                "Would you like to know about the fund's expense ratio?"
            ],
            "tax": [
                "Would you like to explore tax-loss harvesting opportunities?",
                "Are you interested in tax-efficient investment strategies?",
                "Would you like to learn about tax implications for different account types?"
            ],
            # Default questions if no keywords match
            "default": [
                "Would you like to learn more about this topic?",
                "Do you have any specific questions about your investments?",
                "Would you like to explore other investment opportunities?"
            ]
        }
    
    def generate_questions(self, agent_state: Dict[str, Any], 
                          original_response: Dict[str, Any],
                          max_questions: int = 3) -> List[str]:
        """Generate follow-up questions based on keywords in response.
        
        Args:
            agent_state: The current agent state
            original_response: The response from primary agent
            max_questions: Maximum number of questions to return
            
        Returns:
            List of follow-up questions
        """
        # Convert response to string for keyword matching
        response_text = json.dumps(original_response).lower()
        agent_type = agent_state.get("current_agent_type", "").lower()
        
        # Get questions based on matched keywords
        questions = []
        
        # First try to match based on agent type
        if agent_type in self.keyword_mapping:
            questions.extend(self.keyword_mapping[agent_type])
        
        # Then try to match based on content keywords
        for keyword, keyword_questions in self.keyword_mapping.items():
            if keyword in response_text and keyword != "default":
                questions.extend(keyword_questions)
        
        # If no matches, use default questions
        if not questions and "default" in self.keyword_mapping:
            questions = self.keyword_mapping["default"]
        
        # Filter for uniqueness
        unique_questions = []
        for q in questions:
            if q not in unique_questions:
                unique_questions.append(q)
        
        # Limit to max_questions
        return unique_questions[:max_questions]


class FollowUpAgent(AgentBase):
    """Agent that generates contextual follow-up questions based on primary agent responses."""
    
    def __init__(self, 
                generator_strategy: FollowUpGenerator,
                max_questions: int = 3):
        """Initialize the Follow-Up Agent.
        
        Args:
            generator_strategy: Strategy to use for generating follow-up questions
            max_questions: Maximum number of follow-up questions to generate
        """
        super().__init__()
        self.generator_strategy = generator_strategy
        self.max_questions = max_questions
    
    def process(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent state and generate follow-up questions.
        
        Args:
            agent_state: The current agent state with primary agent's response
            
        Returns:
            Updated agent state with follow-up questions added
        """
        logger.info("FollowUpAgent processing started")
        
        # Get the original response from the primary agent
        if "primary_agent_response" not in agent_state:
            logger.warning("No primary agent response found in agent state")
            return agent_state  # Return unchanged if no primary response
        
        original_response = agent_state["primary_agent_response"]
        
        # Generate follow-up questions using the strategy pattern
        follow_up_questions = self.generator_strategy.generate_questions(
            agent_state=agent_state,
            original_response=original_response,
            max_questions=self.max_questions
        )
        
        # Add follow-up questions to agent state
        agent_state["follow_up_questions"] = follow_up_questions
        
        logger.info(f"Generated {len(follow_up_questions)} follow-up questions")
        return agent_state
    
    @classmethod
    def create_with_llm(cls, llm_handler: LLMHandler, max_questions: int = 3) -> 'FollowUpAgent':
        """Factory method to create agent with LLM-based strategy.
        
        Args:
            llm_handler: Handler for LLM API calls
            max_questions: Maximum number of follow-up questions
            
        Returns:
            Configured FollowUpAgent instance
        """
        generator = LLMBasedFollowUpGenerator(llm_handler)
        return cls(generator, max_questions)
    
    @classmethod
    def create_for_testing(cls, max_questions: int = 3) -> 'FollowUpAgent':
        """Factory method to create agent with keyword-based strategy for testing.
        
        Args:
            max_questions: Maximum number of follow-up questions
            
        Returns:
            FollowUpAgent configured for testing
        """
        generator = KeywordBasedFollowUpGenerator()
        return cls(generator, max_questions)
