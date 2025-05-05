from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

import logging

from services.agents.base.agent import BaseAgent
from services.agents.base.state import AgentState
from services.llm_service.llm_chat_service import LLMChatService
from services.llm_service.model_enum import ModelProviderEnum

logger = logging.getLogger(__name__)

class FollowUpAgent(BaseAgent):
    """
    Agent responsible for following up with users to gather additional information
    when initial input is insufficient or requires clarification.
    """
    def __init__(self):
        super().__init__()
        self.agent_name = "FollowUp"
        # Define the prompt template to guide the agent's follow-up behavior
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
        self.llm_service = LLMChatService(ModelProviderEnum.OPENAI_MODEL)
    
    def execute_agent(self, state: AgentState) -> AgentState:
        """
        Generate appropriate follow-up questions based on the user's query and current state.
        
        Args:
            state: The current agent state containing user query and context
            
        Returns:
            Updated agent state with follow-up questions and additional context
        """
        last_message = state.agent_request.query
        conversation_history = self._format_conversation_history(state)
        
        # Identify information gaps and generate follow-up questions
        follow_up_result = self._generate_follow_up_questions(
            query=last_message,
            conversation_history=conversation_history,
            current_context=self._extract_context(state)
        )
        
        # Update state with follow-up information
        state.follow_up_data = follow_up_result.get("data", {})
        state.agent_response = follow_up_result.get("response", "")
        
        logger.info(f"Follow-up questions generated: {state.agent_response}")
        return state
    
    def _format_conversation_history(self, state: AgentState) -> List[Dict]:
        """
        Extracts and formats the conversation history from the agent state.
        
        Args:
            state: The current agent state
            
        Returns:
            List of formatted conversation messages
        """
        history = []
        if hasattr(state, 'conversation_history') and state.conversation_history:
            for exchange in state.conversation_history:
                if 'user' in exchange:
                    history.append({"role": "user", "content": exchange['user']})
                if 'assistant' in exchange:
                    history.append({"role": "assistant", "content": exchange['assistant']})
        return history
    
    def _extract_context(self, state: AgentState) -> Dict:
        """
        Extracts relevant context from the agent state for follow-up generation.
        
        Args:
            state: The current agent state
            
        Returns:
            Dictionary containing context information
        """
        context = {}
        
        # Extract fund information if available
        if hasattr(state, 'fund_data') and state.fund_data:
            context['fund_information'] = state.fund_data
        
        # Extract user profile information if available
        if hasattr(state, 'user_profile') and state.user_profile:
            context['user_profile'] = state.user_profile
        
        # Extract classification information if available
        if hasattr(state, 'query_classification') and state.query_classification:
            context['classification'] = state.query_classification
        
        return context
    
    def _generate_follow_up_questions(self, query: str, conversation_history: List[Dict], current_context: Dict) -> Dict:
        """
        Generates follow-up questions based on the user query and available context.
        
        Args:
            query: The user's last query
            conversation_history: List of previous conversation exchanges
            current_context: Dictionary containing relevant context information
            
        Returns:
            Dictionary containing follow-up data and response
        """
        # Prepare inputs for LLM
        inputs = {
            "input": query,
            "messages": conversation_history,
            "context": current_context
        }
        
        # Generate follow-up questions using the LLM
        llm_chain = self.prompt | self.llm_service.chat_model
        response = llm_chain.invoke(inputs)
        
        # Extract follow-up questions from the response
        follow_up_questions = self._parse_follow_up_questions(response.content)
        
        logger.info(f"Generated follow-up questions: {follow_up_questions}")
        
        return {
            "data": {
                "missing_information": follow_up_questions.get("missing_info", []),
                "clarification_needed": follow_up_questions.get("clarification_needed", []),
                "follow_up_questions": follow_up_questions.get("questions", [])
            },
            "response": response.content
        }
    
    def _parse_follow_up_questions(self, response_content: str) -> Dict:
        """
        Parses the LLM response to extract structured follow-up question data.
        
        Args:
            response_content: The raw response from the LLM
            
        Returns:
            Dictionary containing parsed follow-up information
        """
        # This is a simple implementation that could be enhanced with more 
        # sophisticated parsing logic in the future
        lines = response_content.strip().split('\n')
        questions = [line.strip() for line in lines if '?' in line]
        
        return {
            "questions": questions,
            "missing_info": self._identify_missing_information(response_content),
            "clarification_needed": self._identify_clarification_needed(response_content)
        }
    
    def _identify_missing_information(self, response_content: str) -> List[str]:
        """
        Identifies missing information categories from the response.
        
        Args:
            response_content: The raw response from the LLM
            
        Returns:
            List of missing information categories
        """
        missing_info = []
        
        # Common information categories in financial/investment queries
        information_categories = [
            "timeframe", "fund name", "investment amount", 
            "risk profile", "investment goals", "portfolio details"
        ]
        
        # Check for mentions of missing information
        for category in information_categories:
            if f"missing {category}" in response_content.lower() or \
               f"need {category}" in response_content.lower() or \
               f"specify {category}" in response_content.lower():
                missing_info.append(category)
        
        return missing_info
    
    def _identify_clarification_needed(self, response_content: str) -> List[str]:
        """
        Identifies areas that need clarification from the response.
        
        Args:
            response_content: The raw response from the LLM
            
        Returns:
            List of areas needing clarification
        """
        clarification_areas = []
        
        # Common clarification areas in financial/investment queries
        clarification_categories = [
            "comparison criteria", "performance metrics", 
            "specific funds", "investment strategy",
            "target returns", "risk tolerance"
        ]
        
        # Check for mentions of clarification needs
        for category in clarification_categories:
            if f"clarify {category}" in response_content.lower() or \
               f"unclear {category}" in response_content.lower() or \
               f"ambiguous {category}" in response_content.lower():
                clarification_areas.append(category)
        
        return clarification_areas
