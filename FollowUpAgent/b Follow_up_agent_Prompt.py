# follow_up_agent_prompt.py
""" 
Prompt generation and response parsing for the Follow-Up Agent.

This module contains functions to generate prompt templates for the LLM and 
parse responses to extract relevant follow-up questions.
"""

import json
import re
from typing import Dict, List, Any, Optional, Union


def generate_follow_up_prompt(
    user_query: str,
    agent_response: Dict[str, Any],
    agent_type: str = "",
    user_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_questions: int = 3
) -> str:
    """Generate a prompt for the LLM to create follow-up questions.
    
    Args:
        user_query: The original user query
        agent_response: The response from the primary agent
        agent_type: Type of the primary agent (portfolio, fund, tax, etc.)
        user_context: Optional user context information
        conversation_history: Optional conversation history
        max_questions: Maximum number of questions to generate
        
    Returns:
        Formatted prompt string for the LLM
    """
    # Convert agent response to a readable format
    response_text = ""
    if isinstance(agent_response, dict):
        # Extract the main content from response based on common response structures
        if "message" in agent_response:
            response_text = agent_response["message"]
        elif "content" in agent_response:
            response_text = agent_response["content"]
        elif "response" in agent_response:
            response_text = agent_response["response"]
        else:
            # Try to convert the whole response to text if no standard field is found
            try:
                response_text = json.dumps(agent_response, indent=2)
            except:
                response_text = str(agent_response)
    else:
        response_text = str(agent_response)
    
    # Extract previous questions to avoid duplication
    previous_questions = []
    if conversation_history:
        for entry in conversation_history:
            if "follow_up_questions" in entry:
                previous_questions.extend(entry["follow_up_questions"])
            # Also check for user queries to avoid suggesting already asked questions
            if "user_query" in entry:
                previous_questions.append(entry["user_query"])
    
    # Format user context if available
    context_text = ""
    if user_context:
        context_text = "User Context:\n"
        for key, value in user_context.items():
            context_text += f"- {key}: {value}\n"
    
    # Build the prompt
    prompt = f"""
You are an intelligent follow-up question generator for a financial advisory system. 
Your task is to generate {max_questions} relevant and engaging follow-up questions based on the 
user's original query and the agent's response.

User's Original Query: 
{user_query}

Agent Type: {agent_type if agent_type else "Not specified"}

Agent's Response:
{response_text}

{context_text if context_text else ""}

REQUIREMENTS FOR FOLLOW-UP QUESTIONS:
1. Generate exactly {max_questions} follow-up questions that naturally extend the conversation.
2. Questions should be relevant to the user's original query and the agent's response.
3. Questions should encourage deeper exploration of the topic or related financial concepts.
4. Questions should be concise and conversational (15 words or less).
5. Questions should not duplicate any previously asked questions.
6. Format your response as a JSON array of strings only.

Previously asked questions (DO NOT DUPLICATE THESE):
{json.dumps(previous_questions, indent=2) if previous_questions else "None"}

OUTPUT FORMAT EXAMPLE:
[
  "Question 1?",
  "Question 2?",
  "Question 3?"
]
"""
    return prompt


def parse_follow_up_response(llm_response: str) -> List[str]:
    """Parse the LLM response to extract follow-up questions.
    
    Args:
        llm_response: Raw response from the LLM
        
    Returns:
        List of follow-up questions
    """
    # First try to parse as JSON
    try:
        # Find JSON array in the response using regex
        json_match = re.search(r'\[[\s\S]*\]', llm_response)
        if json_match:
            questions = json.loads(json_match.group(0))
            
            # Validate that it's a list of strings
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
    except json.JSONDecodeError:
        pass  # Fall back to regex parsing
    
    # Fallback: Extract questions using regex patterns
    questions = []
    
    # Pattern 1: Look for numbered questions
    numbered_pattern = r'(?:^|\n)(?:\d+[\.\)]\s*|["\'*-]\s*)(.*?\?)'
    matches = re.findall(numbered_pattern, llm_response)
    if matches:
        questions.extend(matches)
    
    # Pattern 2: Look for any sentence ending with question mark
    if not questions:
        question_pattern = r'([A-Z][^.!?]*?\?)'
        matches = re.findall(question_pattern, llm_response)
        questions.extend(matches)
    
    # Clean up questions
    clean_questions = []
    for q in questions:
        # Remove quotes and leading/trailing whitespace
        cleaned = q.strip('"\'').strip()
        if cleaned and cleaned not in clean_questions:
            clean_questions.append(cleaned)
    
    return clean_questions
