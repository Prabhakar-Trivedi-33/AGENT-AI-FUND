import logging
from typing import Dict, List, Any
from services.agents.base.agent import BaseAgent
from services.agents.base.state import AgentState
from services.llm_service.llm_chat_service import LLMChatService
from services.llm_service.model_enum import ModelProviderEnum
from services.agents.dtos.agent_system_dtos import AgentSystemRequest
from prompt_repository.system.system_prompts import ARTH_SYSTEM_PROMPT
from prompt_repository.agents.arth_agent_prompts import USER_PORTFOLIO_SCHEMA, MUTUAL_FUND_CATEGORY_INFO_SCHEMA
import json

logger = logging.getLogger(__name__)

# --- Response Parsing Strategy ---
class FollowUpResponseParser:
    """
    Strategy for parsing and validating LLM responses for follow-up questions.
    """
    @staticmethod
    def parse(response: str) -> Dict[str, Any]:
        """
        Parse the LLM response and return a dict with keys:
        - 'questions': List[str] of follow-up questions (if any)
        - 'no_follow_up_needed': bool
        """
        response = response.strip()
        if response.lower().startswith("no follow-up needed"):
            return {"questions": [], "no_follow_up_needed": True}
        try:
            # Try to parse as JSON
            data = json.loads(response)
            if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
                questions = [q for q in data["questions"] if isinstance(q, str) and q.strip()]
                return {"questions": questions, "no_follow_up_needed": False}
            elif isinstance(data, list):
                questions = [q for q in data if isinstance(q, str) and q.strip()]
                return {"questions": questions, "no_follow_up_needed": False}
        except Exception:
            # Fallback: try to extract questions from numbered or bulleted list
            lines = [l.strip("- ") for l in response.splitlines() if l.strip()]
            questions = [l for l in lines if l.endswith("?")]
            if questions:
                return {"questions": questions, "no_follow_up_needed": False}
        # If nothing found, treat as no follow-up needed
        return {"questions": [], "no_follow_up_needed": True}

class FollowUpAgent(BaseAgent):
    """
    Agent responsible for generating 3-5 targeted follow-up questions to gather more information from the user.
    Uses LLM to generate high-impact, context-aware follow-up questions in a structured format.
    """
    MAX_HISTORY_CHARS = 2000  # Soft limit for history
    MAX_PROMPT_CHARS = 10000  # Hard limit for the entire prompt

    def __init__(self):
        super().__init__()
        self.agent_name = "FollowUpAgent"
        self.llm_service = LLMChatService(ModelProviderEnum.OPEN_AI_MODEL)

    def _truncate_history_to_fit_prompt(self, static_prompt: str, history: str) -> str:
        """
        Truncate the history so that the total prompt length does not exceed MAX_PROMPT_CHARS.
        """
        available = self.MAX_PROMPT_CHARS - len(static_prompt)
        if available <= 0:
            return ""
        if len(history) > available:
            return history[-available:]
        return history

    def _build_followup_prompt(self, agent_request: AgentSystemRequest) -> str:
        """
        Build a prompt for the LLM to generate 3-5 follow-up questions in JSON format.
        Includes relevant data schemas for better context.
        Ensures the total prompt length does not exceed MAX_PROMPT_CHARS.
        """
        static_prompt = f"""
{ARTH_SYSTEM_PROMPT}

# Data Schemas for Reference
{USER_PORTFOLIO_SCHEMA}
{MUTUAL_FUND_CATEGORY_INFO_SCHEMA}

You are a follow-up agent. Your task is to ask 3-5 targeted, high-impact follow-up questions to clarify the user's intent or gather missing information.

## User Chat History:
"""
        truncated_history = self._truncate_history_to_fit_prompt(static_prompt, agent_request.history)
        prompt = (
            static_prompt
            + truncated_history
            + f"""

## User Query:
{agent_request.query}

## Instructions:
- Ask only for information that is necessary to proceed.
- Be concise and clear.
- If the query is already clear, respond with 'No follow-up needed.'
- Respond ONLY in the following JSON format:
{{
  "questions": [
    "First follow-up question?",
    "Second follow-up question?",
    ...
  ]
}}
- If no follow-up is needed, respond with the string: No follow-up needed.
"""
        )
        # Final hard cut if needed
        if len(prompt) > self.MAX_PROMPT_CHARS:
            prompt = prompt[:self.MAX_PROMPT_CHARS]
        return prompt

    def execute_agent(self, state: AgentState) -> AgentState:
        """
        Generate 3-5 follow-up questions using LLM and update the agent_response in state.
        Handles response parsing, validation, and special cases.
        """
        try:
            logger.info(f"{self.agent_name}: Building follow-up prompt.")
            prompt = self._build_followup_prompt(state.agent_request)
            logger.info(f"{self.agent_name}: Sending prompt to LLM.")
            followup_response = self.llm_service.chat(prompt)
            logger.info(f"{self.agent_name}: LLM follow-up response: {followup_response}")
            parsed = FollowUpResponseParser.parse(followup_response)
            if parsed["no_follow_up_needed"]:
                state.agent_response["follow_up_questions"] = []
                state.agent_response["follow_up_needed"] = False
                state.agent_response["message"] = "No follow-up needed."
            else:
                state.agent_response["follow_up_questions"] = parsed["questions"]
                state.agent_response["follow_up_needed"] = True
                state.agent_response["message"] = "Follow-up questions generated."
        except Exception as e:
            logger.error(f"{self.agent_name}: Error generating follow-up questions: {e}")
            state.agent_response["follow_up_questions"] = []
            state.agent_response["follow_up_needed"] = False
            state.agent_response["message"] = "Sorry, I couldn't generate follow-up questions at this time."
        return state
    
