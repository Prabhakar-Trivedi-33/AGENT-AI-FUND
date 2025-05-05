from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from services.agents.base.agent import BaseAgent
from services.agents.base.state import AgentState
from services.llm_service.llm_chat_service import LLMChatService
from services.llm_service.model_enum import ModelProviderEnum
from prompt_repository.agents.follow_up_agent_prompts import (
    FOLLOW_UP_AGENT_SYSTEM_PROMPT, 
    generate_follow_up_questions
)
import json
import logging

logger = logging.getLogger(__name__)

class FollowUpAgent(BaseAgent):
    def __init__(self):
        """
        Initialize the Follow-Up Agent with advanced configuration
        """
        super().__init__()
        self.agent_name = "FollowUp"
        
        # Advanced prompt template for follow-up interactions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", FOLLOW_UP_AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}")
        ])
        
        # Use OpenAI model for follow-up question generation
        self.llm_service = LLMChatService(ModelProviderEnum.OPENAI_MODEL)

    def execute_agent(self, state: AgentState) -> AgentState:
        """
        Execute follow-up agent to generate clarifying questions and extract context
        
        :param state: Current conversation state
        :return: Updated conversation state with follow-up insights
        """
        try:
            # Prepare comprehensive context for follow-up
            context = self._prepare_follow_up_context(state)
            
            # Generate follow-up questions
            follow_up_result = self._generate_follow_up_questions(context)
            
            # Update state with follow-up information
            if follow_up_result.get("follow_up_questions"):
                state.follow_up_questions = follow_up_result["follow_up_questions"]
                state.follow_up_reasoning = follow_up_result.get("reasoning", "")
                state.follow_up_confidence = follow_up_result.get("confidence_score", 0.5)
            
            # Log follow-up generation details
            logger.info(f"Follow-up generated: {json.dumps(follow_up_result, indent=2)}")
            
            return state
        
        except Exception as e:
            # Comprehensive error handling
            logger.error(f"Follow-up agent error: {e}")
            state.error = str(e)
            return state

    def _prepare_follow_up_context(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare a comprehensive context for follow-up question generation
        
        :param state: Current agent state
        :return: Detailed context dictionary
        """
        context = {
            "user_query": state.agent_request.query,
            "conversation_history": [
                msg for msg in state.agent_request.messages 
                if hasattr(msg, 'content')
            ],
            "completed_agents": state.completed_agents,
            "agent_response": state.agent_response,
            "current_intent": self._extract_intent(state),
            "additional_context": self._extract_additional_context(state)
        }
        return context

    def _extract_intent(self, state: AgentState) -> Optional[str]:
        """
        Extract the primary intent from the current state
        
        :param state: Current agent state
        :return: Extracted intent or None
        """
        try:
            # Check if intent is already defined in agent response
            if state.agent_response and 'intent' in state.agent_response:
                return state.agent_response['intent']
            
            # Fallback: try to infer intent from last query
            return self._infer_intent_from_query(state.agent_request.query)
        
        except Exception as e:
            logger.warning(f"Intent extraction error: {e}")
            return None

    def _infer_intent_from_query(self, query: str) -> Optional[str]:
        """
        Use LLM to infer intent from user query
        
        :param query: User's query
        :return: Inferred intent
        """
        try:
            # Use LLM to classify intent
            intent_prompt = f"""
            Classify the intent of the following query in one word or short phrase:
            Query: {query}
            
            Possible intents: 
            - Information Seeking
            - Clarification
            - Recommendation
            - Comparison
            - Problem Solving
            """
            
            response = self.llm_service.generate_response(intent_prompt)
            return response.strip()
        
        except Exception as e:
            logger.warning(f"Intent inference error: {e}")
            return None

    def _extract_additional_context(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract additional contextual information from the state
        
        :param state: Current agent state
        :return: Additional context dictionary
        """
        additional_context = {}
        
        # Extract relevant contextual information from state
        contextual_keys = [
            'fund_data', 
            'portfolio_data', 
            'user_preferences', 
            'investment_goals'
        ]
        
        for key in contextual_keys:
            if hasattr(state, key):
                additional_context[key] = getattr(state, key)
        
        return additional_context

    def _generate_follow_up_questions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate follow-up questions using a custom generation function
        
        :param context: Comprehensive conversation context
        :return: Follow-up questions with reasoning
        """
        try:
            # Use the custom follow-up question generation function
            follow_up_result = generate_follow_up_questions(context)
            
            # Validate generated questions
            if not self._validate_follow_up_questions(follow_up_result.get("follow_up_questions", [])):
                # Fallback to LLM-based generation if validation fails
                follow_up_result = self._llm_generate_follow_up_questions(context)
            
            return follow_up_result
        
        except Exception as e:
            logger.error(f"Follow-up question generation error: {e}")
            return {
                "follow_up_questions": [],
                "reasoning": f"Error in question generation: {e}",
                "confidence_score": 0.0
            }

    def _llm_generate_follow_up_questions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback method to generate follow-up questions using LLM
        
        :param context: Comprehensive conversation context
        :return: Follow-up questions with reasoning
        """
        try:
            prompt = f"""
            Based on the following context, generate 2-3 precise follow-up questions:
            
            Context: {json.dumps(context, indent=2)}
            
            Guidelines:
            - Ask questions that clarify or expand on the current conversation
            - Focus on extracting additional meaningful information
            - Ensure questions are specific and relevant
            """
            
            response = self.llm_service.generate_response(prompt)
            
            # Parse LLM response into structured format
            follow_up_questions = [
                q.strip() for q in response.split('\n') 
                if q.strip() and q.strip().endswith('?')
            ][:3]
            
            return {
                "follow_up_questions": follow_up_questions,
                "reasoning": "LLM-generated follow-up questions",
                "confidence_score": 0.7
            }
        
        except Exception as e:
            logger.error(f"LLM follow-up generation error: {e}")
            return {
                "follow_up_questions": [],
                "reasoning": f"LLM generation failed: {e}",
                "confidence_score": 0.0
            }

    def _validate_follow_up_questions(self, questions: List[str]) -> bool:
        """
        Validate generated follow-up questions
        
        :param questions: List of follow-up questions
        :return: Whether questions meet quality criteria
        """
        if not questions:
            return False
        
        # Validation criteria
        criteria = [
            lambda q: len(q) > 10,  # Minimum meaningful length
            lambda q: q.endswith('?'),  # Must be a question
            lambda q: len(questions) <= 3  # Maximum 3 questions
        ]
        
        return all(all(check(q) for check in criteria) for q in questions)
    
