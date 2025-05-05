import pytest
from services.database_manager.sql_database import get_sql_session
from services.database_manager.sql_curd import fetch_data
from services.database_manager.database_enum import DatabaseFormat
import logging
import json

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def db_session():
    """Fixture to get a database session for testing"""
    session = next(get_sql_session())
    try:
        yield session
    finally:
        session.close()

def test_fetch_data_basic_functionality(db_session):
    """Test basic data fetching functionality"""
    result = fetch_data(
        db=db_session,
        table_name="mutual_fund_nav",
        filters={},  # No filters to get some data
        columns=["scheme_nav_date", "scheme_nav_value"],
        output_format=DatabaseFormat.JSON
    )
    
    # Check that the result is a valid JSON string
    assert result is not None, "Fetch data returned None"
    
    try:
        parsed_result = json.loads(result)
    except json.JSONDecodeError:
        pytest.fail("Returned data is not a valid JSON string")
    
    # Optionally, add more specific assertions based on your data model
    assert isinstance(parsed_result, list), "Result should be a list"

def test_fetch_data_with_specific_filters(db_session):
    """Test fetching data with specific filters"""
    result = fetch_data(
        db=db_session,
        table_name="mutual_fund_nav",
        filters={
            "scheme_code": {
                "operator": "in",
                "value": ["119001", "119002", "119003", "119004", "119005"]
            },
        },
        columns=["scheme_nav_date", "scheme_nav_value"],
        output_format=DatabaseFormat.JSON
    )
    
    # Parse the result
    try:
        parsed_result = json.loads(result)
    except json.JSONDecodeError:
        pytest.fail("Returned data is not a valid JSON string")
    
    # You might want to adjust these assertions based on your actual data
    assert isinstance(parsed_result, list), "Result should be a list"

def test_fetch_data_invalid_table(db_session):
    """Test error handling for invalid table name"""
    with pytest.raises(Exception) as excinfo:
        fetch_data(
            db=db_session,
            table_name="non_existent_table",
            filters={},
            columns=["some_column"],
            output_format=DatabaseFormat.JSON
        )
    
    # Optionally check the specific type of exception
    assert "does not exist" in str(excinfo.value).lower()

def test_fetch_data_invalid_columns(db_session):
    """Test error handling for invalid columns"""
    with pytest.raises(Exception) as excinfo:
        fetch_data(
            db=db_session,
            table_name="mutual_fund_nav",
            filters={},
            columns=["non_existent_column"],
            output_format=DatabaseFormat.JSON
        )
    
    # Optionally check the specific type of exception
    assert "column" in str(excinfo.value).lower() and "not found" in str(excinfo.value).lower()
