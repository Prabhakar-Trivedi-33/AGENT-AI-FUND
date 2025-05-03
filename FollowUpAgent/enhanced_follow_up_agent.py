from typing import Dict, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from services.agents.base.agent import BaseAgent
from services.agents.base.state import AgentState
from services.database_manager.database_enum import DatabaseFormat
from services.database_manager.sql_curd import get_mutual_fund_info, get_fund_category_info, get_mutual_fund_index_nav
from services.database_manager.sql_database import get_sql_session
from services.llm_service.llm_chat_service import LLMChatService
from services.llm_service.model_enum import ModelProviderEnum
import re
import logging

logger = logging.getLogger(__name__)

# Define enums for query classification
class QueryClassification:
    GENERAL_FUNDS_INFORMATION = "GENERAL_FUNDS_INFORMATION"
    GENERAL_INFORMATION = "GENERAL_INFORMATION"
    PERSONALISED_INFORMATION = "PERSONALISED_INFORMATION"

class FundInformationAgent(BaseAgent):
    """
    Agent responsible for providing information about mutual funds.
    Handles three types of queries:
    1. General Fund Information - Specific fund details
    2. General Market Information - Overall market conditions
    3. Personalized Portfolio Information - User-specific portfolio analysis
    """
    def __init__(self):
        super().__init__()
        self.agent_name = "FundInfo"
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Fund Information agent. Provide detailed information about:
            - Current market conditions
            - Fund performance and recommendations
            - Risk analysis
            - Investment opportunities
            
            Focus only on market and fund related information. Be concise but comprehensive.
            Structure your response with clear headings and bullet points where appropriate.
            When discussing funds, include important metrics like:
            - Current NAV
            - Historical performance (1yr, 3yr, 5yr)
            - Risk metrics
            - Category comparison
            """),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}")
        ])
        # Using Gemini for fund analysis as mentioned in the original code
        self.llm_service = LLMChatService(ModelProviderEnum.GOOGLE_GEMINI_MODEL)
        
        # LLM for query classification - using a faster model for this task
        self.classification_llm = LLMChatService(ModelProviderEnum.GOOGLE_GEMINI_MODEL)
        
        # Classification prompt
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier. Classify the user's query into exactly one of these categories:
            
            - GENERAL_FUNDS_INFORMATION: Queries about specific mutual funds, their performance, details, etc.
              Example: "Tell me about ICICI Prudential Balanced Advantage Fund"
              
            - GENERAL_INFORMATION: Queries about market trends, general investment advice, etc.
              Example: "How is the market performing today?"
              
            - PERSONALISED_INFORMATION: Queries about the user's own portfolio or personalized recommendations.
              Example: "How is my portfolio performing?" or "What funds should I invest in based on my risk profile?"
            
            Respond with only the category name, nothing else."""),
            ("human", "{query}")
        ])

    def execute_agent(self, state: AgentState) -> AgentState:
        """
        Main execution flow of the Fund Information Agent.
        
        Args:
            state: Current agent state containing conversation history and data
            
        Returns:
            Updated agent state with fund information and response
        """
        messages = state.messages
        last_message = messages[-1].content
        
        # Step 1: Classify the query to determine appropriate action
        query_type = self._classify_query(last_message)
        logger.info(f"Query classified as: {query_type}")
        
        # Step 2: Process based on classification
        if query_type == QueryClassification.GENERAL_FUNDS_INFORMATION:
            fund_analysis = self._handle_specific_fund_query(last_message, state)
        elif query_type == QueryClassification.PERSONALISED_INFORMATION:
            fund_analysis = self._handle_personalized_query(last_message, state)
        else:  # GENERAL_INFORMATION
            fund_analysis = self._handle_general_market_query(last_message)
        
        # Step 3: Update state with fund data and response
        if "data" in fund_analysis:
            state.fund_data.update(fund_analysis["data"])
        
        messages.append(AIMessage(content=fund_analysis["response"]))
        
        return state
    
    def _classify_query(self, query: str) -> str:
        """
        Classifies the user query to determine the type of information requested.
        
        Args:
            query: User's query string
            
        Returns:
            Classification category as defined in QueryClassification
        """
        # Pattern matching for quick classification without LLM when possible
        if any(pattern in query.lower() for pattern in ["my portfolio", "my investment", "my fund", "i invested"]):
            return QueryClassification.PERSONALISED_INFORMATION
        
        if any(pattern in query.lower() for pattern in ["market", "trend", "economy", "index", "sensex", "nifty"]):
            return QueryClassification.GENERAL_INFORMATION
        
        # For more complex queries, use LLM classification
        try:
            classification_chain = self.classification_prompt | self.classification_llm
            result = classification_chain.invoke({"query": query})
            
            # Extract classification from result
            if QueryClassification.GENERAL_FUNDS_INFORMATION in result:
                return QueryClassification.GENERAL_FUNDS_INFORMATION
            elif QueryClassification.PERSONALISED_INFORMATION in result:
                return QueryClassification.PERSONALISED_INFORMATION
            elif QueryClassification.GENERAL_INFORMATION in result:
                return QueryClassification.GENERAL_INFORMATION
            else:
                # Default to general information if classification is unclear
                logger.warning(f"Query classification unclear: {result}. Defaulting to GENERAL_INFORMATION")
                return QueryClassification.GENERAL_INFORMATION
        except Exception as e:
            logger.error(f"Error during query classification: {e}")
            return QueryClassification.GENERAL_INFORMATION
    
    def _extract_fund_names(self, query: str) -> List[str]:
        """
        Extracts potential fund names from the user query.
        
        Args:
            query: User's query string
            
        Returns:
            List of potential fund names
        """
        # Define a prompt to extract fund names
        fund_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract all mutual fund names mentioned in the query. 
            Return only the fund names as a comma-separated list. 
            If no fund names are found, return "None".
            Example: "Tell me about ICICI Prudential Balanced Advantage Fund and HDFC Top 100 Fund" 
            → "ICICI Prudential Balanced Advantage Fund, HDFC Top 100 Fund"
            """),
            ("human", "{query}")
        ])
        
        try:
            # Run extraction chain
            extraction_chain = fund_extraction_prompt | self.llm_service
            result = extraction_chain.invoke({"query": query})
            
            # Parse results
            if "None" in result:
                return []
            
            fund_names = [name.strip() for name in result.split(",")]
            return fund_names
        except Exception as e:
            logger.error(f"Error extracting fund names: {e}")
            return []
    
    def _get_scheme_codes_from_names(self, fund_names: List[str]) -> List[str]:
        """
        Retrieves scheme codes for given fund names using vector search.
        
        Args:
            fund_names: List of fund names to look up
            
        Returns:
            List of scheme codes
        """
        # TODO: Implement vector search for fund names
        # This would typically involve:
        # 1. Query vector database with fund names
        # 2. Return matching scheme codes
        
        # For now, using mock implementation
        # In a real implementation, this would query a vector database
        mock_scheme_mapping = {
            "icici prudential balanced advantage fund": "21521",
            "hdfc top 100 fund": "21651",
            "axis bluechip fund": "19575",
            "sbi small cap fund": "35511",
            "aditya birla sun life tax relief 96": "43310"
        }
        
        scheme_codes = []
        for fund_name in fund_names:
            normalized_name = fund_name.lower()
            for key in mock_scheme_mapping:
                if key in normalized_name or normalized_name in key:
                    scheme_codes.append(mock_scheme_mapping[key])
                    break
        
        # If no matches found, return some default funds for testing
        if not scheme_codes:
            logger.warning(f"No scheme codes found for fund names: {fund_names}. Using defaults.")
            scheme_codes = ["21521", "21651", "19575"]
        
        return scheme_codes
    
    def _handle_specific_fund_query(self, query: str, state: AgentState) -> Dict:
        """
        Handles queries about specific mutual funds.
        
        Args:
            query: User's query string
            state: Current agent state
            
        Returns:
            Dictionary with fund data and formatted response
        """
        # Step 1: Extract fund names from query
        fund_names = self._extract_fund_names(query)
        logger.info(f"Extracted fund names: {fund_names}")
        
        # Step 2: Get scheme codes for the fund names
        scheme_codes = self._get_scheme_codes_from_names(fund_names)
        logger.info(f"Mapped to scheme codes: {scheme_codes}")
        
        # Step 3: Fetch fund information from database
        with get_sql_session() as session:
            # Get fund information
            fund_info = get_mutual_fund_info(session, scheme_codes, DatabaseFormat.CSV)
            
            # Get category information for these funds
            # In a real implementation, we would extract category codes from fund_info
            category_codes = ["83", "33", "32"]  # Mock category codes
            fund_category_info = get_fund_category_info(session, category_codes, DatabaseFormat.CSV)
            
            # Get benchmark index NAV for comparison
            fund_index_nav = get_mutual_fund_index_nav(session, category_codes, DatabaseFormat.CSV)
        
        # Step 4: Summarize and format the response using LLM
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mutual fund expert. Create a concise but comprehensive summary of the provided fund information.
            Focus on:
            - Key performance metrics
            - Comparison with category average
            - Risk analysis
            - Notable strengths and weaknesses
            
            Format your response with clear headings and bullet points for readability.
            """),
            ("human", f"""
            FUND INFORMATION:
            {fund_info}
            
            CATEGORY INFORMATION:
            {fund_category_info}
            
            INDEX INFORMATION:
            {fund_index_nav}
            
            Based on this information, provide a comprehensive analysis of these funds.
            """)
        ])
        
        try:
            summary_chain = summary_prompt | self.llm_service
            fund_summary = summary_chain.invoke({})
            
            # Return both raw data and human-readable summary
            return {
                "data": {
                    "fund_info": fund_info,
                    "fund_category_info": fund_category_info,
                    "fund_index_nav": fund_index_nav,
                    "scheme_codes": scheme_codes
                },
                "response": fund_summary
            }
        except Exception as e:
            logger.error(f"Error generating fund summary: {e}")
            # Fallback to basic response if LLM fails
            return {
                "data": {
                    "fund_info": fund_info,
                    "fund_category_info": fund_category_info,
                    "fund_index_nav": fund_index_nav
                },
                "response": self._fund_info_response_format(fund_info, fund_category_info)
            }
    
    def _handle_general_market_query(self, query: str) -> Dict:
        """
        Handles general market information queries.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary with market data and formatted response
        """
        # For general market information, we'll use category indices
        category_codes = ["1", "2", "3", "4", "5"]  # Main market categories
        
        with get_sql_session() as session:
            # Get market indices information
            market_indices = get_mutual_fund_index_nav(session, category_codes, DatabaseFormat.CSV)
            
            # Get some category information for context
            category_info = get_fund_category_info(session, category_codes, DatabaseFormat.CSV)
        
        # Generate market analysis using LLM
        market_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a market analyst. Create a concise summary of current market conditions based on the provided data.
            Focus on:
            - Key market trends
            - Sector performance
            - Notable market movements
            - Economic factors
            
            Format your response with clear headings and bullet points for readability.
            """),
            ("human", f"""
            MARKET INDICES:
            {market_indices}
            
            CATEGORY INFORMATION:
            {category_info}
            
            User Query: {query}
            
            Based on this information, provide a comprehensive market analysis.
            """)
        ])
        
        try:
            market_chain = market_prompt | self.llm_service
            market_analysis = market_chain.invoke({})
            
            return {
                "data": {
                    "market_indices": market_indices,
                    "category_info": category_info
                },
                "response": market_analysis
            }
        except Exception as e:
            logger.error(f"Error generating market analysis: {e}")
            # Fallback response
            return {
                "data": {
                    "market_indices": market_indices,
                    "category_info": category_info
                },
                "response": f"""
                # Market Analysis
                
                ## Key Indices
                {market_indices[:500]}...
                
                ## Category Overview
                {category_info[:500]}...
                
                For more detailed analysis, please try again later.
                """
            }
    
    def _handle_personalized_query(self, query: str, state: AgentState) -> Dict:
        """
        Handles personalized portfolio queries.
        
        Args:
            query: User's query string
            state: Current agent state containing user information
            
        Returns:
            Dictionary with portfolio data and formatted response
        """
        # Get user ID from state
        user_id = state.user_id if hasattr(state, 'user_id') else None
        
        if not user_id:
            return {
                "response": "I don't have access to your portfolio information. Please ensure you're logged in or provide your user ID."
            }
        
        # Fetch user's portfolio data
        with get_sql_session() as session:
            # This would be a custom function to fetch user portfolio
            # For now, mocking the portfolio data structure
            portfolio_data = {
                "investments": [
                    {"scheme_code": "21521", "amount": 50000, "units": 1250.45, "purchase_date": "2023-01-15"},
                    {"scheme_code": "21651", "amount": 75000, "units": 356.78, "purchase_date": "2023-06-22"},
                    {"scheme_code": "19575", "amount": 100000, "units": 2345.67, "purchase_date": "2022-12-05"}
                ],
                "total_investment": 225000,
                "current_value": 248500,
                "gain_loss": 23500,
                "return_percentage": 10.44
            }
            
            # Get fund details for the invested schemes
            scheme_codes = [inv["scheme_code"] for inv in portfolio_data["investments"]]
            fund_info = get_mutual_fund_info(session, scheme_codes, DatabaseFormat.CSV)
        
        # Generate personalized portfolio analysis
        portfolio_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a personal financial advisor. Create a personalized portfolio analysis based on the provided data.
            Focus on:
            - Portfolio performance
            - Asset allocation
            - Risk assessment
            - Recommendations for rebalancing or new investments
            
            Format your response with clear headings and bullet points for readability.
            """),
            ("human", f"""
            USER PORTFOLIO:
            {portfolio_data}
            
            FUND INFORMATION:
            {fund_info}
            
            User Query: {query}
            
            Based on this information, provide a personalized portfolio analysis.
            """)
        ])
        
        try:
            portfolio_chain = portfolio_prompt | self.llm_service
            portfolio_analysis = portfolio_chain.invoke({})
            
            return {
                "data": {
                    "portfolio": portfolio_data,
                    "fund_info": fund_info
                },
                "response": portfolio_analysis
            }
        except Exception as e:
            logger.error(f"Error generating portfolio analysis: {e}")
            # Fallback response
            return {
                "data": {
                    "portfolio": portfolio_data,
                    "fund_info": fund_info
                },
                "response": f"""
                # Your Portfolio Summary
                
                ## Overview
                - Total Investment: ₹{portfolio_data['total_investment']:,}
                - Current Value: ₹{portfolio_data['current_value']:,}
                - Gain/Loss: ₹{portfolio_data['gain_loss']:,} ({portfolio_data['return_percentage']}%)
                
                ## Investments
                You have investments in {len(portfolio_data['investments'])} mutual funds:
                - ICICI Prudential Balanced Advantage Fund: ₹50,000
                - HDFC Top 100 Fund: ₹75,000
                - Axis Bluechip Fund: ₹100,000
                
                For more detailed analysis, please try again later.
                """
            }
    
    def _fund_info_response_format(self, fund_info: str, category_info: str = "") -> str:
        """
        Formats fund information into a readable response when LLM processing fails.
        
        Args:
            fund_info: Raw fund information string
            category_info: Raw category information string
            
        Returns:
            Formatted response string
        """
        # Basic formatting for fallback scenario
        return f"""
        # Mutual Fund Information
        
        ## Fund Details
        {fund_info[:1000]}...
        
        ## Category Information
        {category_info[:500] if category_info else "Category information not available"}
        
        This information is presented in raw format. For a more detailed analysis, please try again later.
        """
