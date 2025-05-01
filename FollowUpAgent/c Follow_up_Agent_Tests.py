# test_follow_up_agent.py
"""
Unit tests for the Follow-Up Agent implementation.
"""

import unittest
from unittest.mock import MagicMock, patch
import json

from .follow_up_agent import (
    FollowUpAgent,
    LLMBasedFollowUpGenerator,
    KeywordBasedFollowUpGenerator
)
from .follow_up_agent_prompt import (
    generate_follow_up_prompt,
    parse_follow_up_response
)


class TestFollowUpAgentPrompt(unittest.TestCase):
    """Tests for prompt generation and response parsing."""
    
    def test_generate_follow_up_prompt(self):
        """Test prompt generation with various inputs."""
        # Test basic prompt generation
        prompt = generate_follow_up_prompt(
            user_query="How is my portfolio performing?",
            agent_response={"message": "Your portfolio is up 5% this year."},
            agent_type="portfolio",
            max_questions=3
        )
        
        # Check that all required elements are in the prompt
        self.assertIn("How is my portfolio performing?", prompt)
        self.assertIn("Your portfolio is up 5% this year.", prompt)
        self.assertIn("portfolio", prompt)
        self.assertIn("3 relevant and engaging follow-up questions", prompt)
        
        # Test with user context
        prompt_with_context = generate_follow_up_prompt(
            user_query="How is my portfolio performing?",
            agent_response={"message": "Your portfolio is up 5% this year."},
            agent_type="portfolio",
            user_context={"risk_level": "moderate", "investment_horizon": "long-term"},
            max_questions=3
        )
        
        # Check that context is included
        self.assertIn("risk_level: moderate", prompt_with_context)
        self.assertIn("investment_horizon: long-term", prompt_with_context)
        
        # Test with conversation history
        history = [
            {"user_query": "Previous question?", 
             "follow_up_questions": ["Already asked question 1?", "Already asked question 2?"]}
        ]
        
        prompt_with_history = generate_follow_up_prompt(
            user_query="How is my portfolio performing?",
            agent_response={"message": "Your portfolio is up 5% this year."},
            agent_type="portfolio",
            conversation_history=history,
            max_questions=3
        )
        
        # Check that previous questions are included to avoid duplication
        self.assertIn("Previous question?", prompt_with_history)
        self.assertIn("Already asked question 1?", prompt_with_history)
        self.assertIn("Already asked question 2?", prompt_with_history)
    
    def test_parse_follow_up_response(self):
        """Test parsing of LLM response to extract questions."""
        # Test parsing valid JSON response
        json_response = '["Question 1?", "Question 2?", "Question 3?"]'
        questions = parse_follow_up_response(json_response)
        self.assertEqual(questions, ["Question 1?", "Question 2?", "Question 3?"])
        
        # Test parsing response with JSON embedded in text
        mixed_response = """
        Here are some follow-up questions:
        
        [
          "Question 1?",
          "Question 2?",
          "Question 3?"
        ]
        """
        questions = parse_follow_up_response(mixed_response)
        self.assertEqual(questions, ["Question 1?", "Question 2?", "Question 3?"])
        
        # Test parsing numbered list fallback
        numbered_response = """
        Here are some follow-up questions:
        
        1. Question 1?
        2. Question 2?
        3. Question 3?
        """
        questions = parse_follow_up_response(numbered_response)
        self.assertEqual(questions, ["Question 1?", "Question 2?", "Question 3?"])
        
        # Test parsing bulleted list fallback
        bulleted_response = """
        Here are some follow-up questions:
        
        - Question 1?
        - Question 2?
        - Question 3?
        """
        questions = parse_follow_up_response(bulleted_response)
        self.assertEqual(questions, ["Question 1?", "Question 2?", "Question 3?"])


class TestFollowUpGenerators(unittest.TestCase):
    """Tests for follow-up question generator strategies."""
    
    def test_llm_based_generator(self):
        """Test LLM-based follow-up generator."""
        # Create mock LLM handler
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '["Question 1?", "Question 2?", "Question 3?"]'
        
        # Create generator with mock
        generator = LLMBasedFollowUpGenerator(mock_llm)
        
        # Test question generation
        agent_state = {
            "user_query": "How is my portfolio performing?",
            "current_agent_type": "portfolio",
            "conversation_history": []
        }
        
        original_response = {"message": "Your portfolio is up 5% this year."}
        
        questions = generator.generate_questions(
            agent_state=agent_state,
            original_response=original_response,
            max_questions=3
        )
        
        # Check that LLM was called with appropriate prompt
        mock_llm.generate_response.assert_called_once()
        prompt = mock_llm.generate_response.call_args[0][0]
        self.assertIn("How is my portfolio performing?", prompt)
        self.assertIn("Your portfolio is up 5% this year.", prompt)
        
        # Check output questions
        self.assertEqual(questions, ["Question 1?", "Question 2?", "Question 3?"])
        
        # Test filtering of duplicate questions
        mock_llm.generate_response.return_value = '["Previous question?", "New question?", "Another new question?"]'
        agent_state_with_history = {
            "user_query": "How is my portfolio performing?",
            "current_agent_type": "portfolio",
            "conversation_history": [
                {"follow_up_questions": ["Previous question?"]}
            ]
        }
        
        questions = generator.generate_questions(
            agent_state=agent_state_with_history,
            original_response=original_response,
            max_questions=3
        )
        
        # Check that duplicate question was filtered out
        self.assertEqual(questions, ["New question?", "Another new question?"])
    
    def test_keyword_based_generator(self):
        """Test keyword-based follow-up generator (mock for testing)."""
        # Create custom keyword mapping
        mapping = {
            "portfolio": [
                "Would you like to see your portfolio details?", 
                "Do you want to know about your asset allocation?"
            ],
            "tax": [
                "Want to learn about tax strategies?", 
                "Interested in tax-loss harvesting?"
            ],
            "default": [
                "Any other questions?", 
                "Would you like more information?"
            ]
        }
        
        # Create generator with custom mapping
        generator = KeywordBasedFollowUpGenerator(mapping)
        
        # Test portfolio-related response
        portfolio_state = {"current_agent_type": "portfolio"}
        portfolio_response = {"message": "Your portfolio information..."}
        
        questions = generator.generate_questions(
            agent_state=portfolio_state,
            original_response=portfolio_response,
            max_questions=2
        )
        
        # Check that portfolio questions were generated
        self.assertEqual(len(questions), 2)
        self.assertIn(questions[0], mapping["portfolio"])
        
        # Test tax-related response
        tax_state = {"current_agent_type": "tax"}
        tax_response = {"message": "Your tax information..."}
        
        questions = generator.generate_questions(
            agent_state=tax_state,
            original_response=tax_response,
            max_questions=2
        )
        
        # Check that tax questions were generated
        self.assertEqual(len(questions), 2)
        self.assertIn(questions[0], mapping["tax"])
        
        # Test default fallback
        unknown_state = {"current_agent_type": "unknown"}
        unknown_response = {"message": "Some information..."}
        
        questions = generator.generate_questions(
            agent_state=unknown_state,
            original_response=unknown_response,
            max_questions=2
        )
        
        # Check that default questions were generated
        self.assertEqual(len(questions), 2)
        self.assertIn(questions[0], mapping["default"])


class TestFollowUpAgent(unittest.TestCase):
    """Tests for the main Follow-Up Agent class."""
    
    def test_process_method(self):
        """Test the main processing method of the agent."""
        # Create mock generator
        mock_generator = MagicMock()
        mock_generator.generate_questions.return_value = [
            "Follow-up question 1?", 
            "Follow-up question 2?"
        ]
        
        # Create agent with mock generator
        agent = FollowUpAgent(mock_generator, max_questions=2)
        
        # Test agent processing
        agent_state = {
            "user_query": "How is my portfolio performing?",
            "primary_agent_response": {"message": "Your portfolio is up 5% this year."}
        }
        
        result_state = agent.process(agent_state)
        
        # Check that generator was called with correct arguments
        mock_generator.generate_questions.assert_called_once_with(
            agent_state=agent_state,
            original_response={"message": "Your portfolio is up 5% this year."},
            max_questions=2
        )
        
        # Check that follow-up questions were added to state
        self.assertIn("follow_up_questions", result_state)
        self.assertEqual(
            result_state["follow_up_questions"], 
            ["Follow-up question 1?", "Follow-up question 2?"]
        )
    
    def test_factory_methods(self):
        """Test factory methods for creating agent instances."""
        # Test LLM-based factory method
        mock_llm = MagicMock()
        llm_agent = FollowUpAgent.create_with_llm(mock_llm, max_questions=3)
        
        # Check that agent was created with correct generator
        self.assertIsInstance(llm_agent, FollowUpAgent)
        self.assertIsInstance(llm_agent.generator_strategy, LLMBasedFollowUpGenerator)
        self.assertEqual(llm_agent.max_questions, 3)
        
        # Test testing factory method
        test_agent = FollowUpAgent.create_for_testing(max_questions=2)
        
        # Check that agent was created with correct generator
        self.assertIsInstance(test_agent, FollowUpAgent)
        self.assertIsInstance(test_agent.generator_strategy, KeywordBasedFollowUpGenerator)
        self.assertEqual(test_agent.max_questions, 2)


if __name__ == "__main__":
    unittest.main()
