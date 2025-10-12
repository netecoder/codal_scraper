"""
Input validation utilities for Codal Scraper
"""

import re
from typing import Any, Union
from datetime import datetime


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class InputValidator:
    """Validates input parameters for Codal queries"""
    
    PATTERNS = {
        'date': r'^[0-9]{4}/(0?[1-9]|1[012])/(0?[1-9]|[12][0-9]|3[01])$',
        'integer_string': r'^[0-9]*$',
        'symbol': r'^[A-Za-zآ-ی0-9]+$',
        'isic': r'^[0-9]{4,6}$'
    }
    
    def __init__(self, value: Any):
        self.value = value
    
    def is_boolean(self) -> bool:
        """Validate if value is boolean"""
        if not isinstance(self.value, bool):
            raise ValidationError(f"Expected boolean, got {type(self.value).__name__}: {self.value}")
        return True
    
    def is_integer(self) -> bool:
        """Validate if value is integer"""
        if not isinstance(self.value, int):
            raise ValidationError(f"Expected integer, got {type(self.value).__name__}: {self.value}")
        return True
    
    def is_string(self) -> bool:
        """Validate if value is string"""
        if not isinstance(self.value, str):
            raise ValidationError(f"Expected string, got {type(self.value).__name__}: {self.value}")
        return True
    
    def is_integer_string(self) -> bool:
        """Validate if value is a string containing only digits"""
        self.is_string()
        if not re.match(self.PATTERNS['integer_string'], str(self.value)):
            raise ValidationError(f"Expected integer string, got: {self.value}")
        return True
    
    def is_date(self, format_str: str = "yyyy/mm/dd") -> bool:
        """Validate if value is a valid date in Persian calendar format"""
        self.is_string()
        if not re.match(self.PATTERNS['date'], str(self.value)):
            raise ValidationError(f"Expected date in format {format_str}, got: {self.value}")
        
        # Additional validation for date components
        parts = str(self.value).split('/')
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        
        if not (1300 <= year <= 1500):
            raise ValidationError(f"Year must be between 1300 and 1500, got: {year}")
        
        if not (1 <= month <= 12):
            raise ValidationError(f"Month must be between 1 and 12, got: {month}")
        
        if not (1 <= day <= 31):
            raise ValidationError(f"Day must be between 1 and 31, got: {day}")
        
        return True
    
    def is_symbol(self) -> bool:
        """Validate if value is a valid stock symbol"""
        self.is_string()
        if not re.match(self.PATTERNS['symbol'], str(self.value)):
            raise ValidationError(f"Invalid symbol format: {self.value}")
        return True
    
    def is_isic(self) -> bool:
        """Validate if value is a valid ISIC code"""
        self.is_string()
        if not re.match(self.PATTERNS['isic'], str(self.value)):
            raise ValidationError(f"Invalid ISIC code format: {self.value}")
        return True
    
    def in_range(self, min_val: Union[int, float], max_val: Union[int, float]) -> bool:
        """Validate if numeric value is within range"""
        if not isinstance(self.value, (int, float)):
            raise ValidationError(f"Expected numeric value, got {type(self.value).__name__}")
        
        if not (min_val <= self.value <= max_val):
            raise ValidationError(f"Value {self.value} is not in range [{min_val}, {max_val}]")
        
        return True
    
    def in_list(self, valid_values: list) -> bool:
        """Validate if value is in a list of valid values"""
        if self.value not in valid_values:
            raise ValidationError(f"Value {self.value} is not in valid values: {valid_values}")
        return True


def validate_input(value: Any, validation_type: str, **kwargs) -> bool:
    """
    Convenience function for input validation
    
    Args:
        value: The value to validate
        validation_type: Type of validation ('boolean', 'integer', 'string', 'date', 'symbol', 'isic', 'range', 'list')
        **kwargs: Additional arguments for specific validation types
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If validation fails
    """
    validator = InputValidator(value)
    
    validation_methods = {
        'boolean': validator.is_boolean,
        'integer': validator.is_integer,
        'string': validator.is_string,
        'integer_string': validator.is_integer_string,
        'date': validator.is_date,
        'symbol': validator.is_symbol,
        'isic': validator.is_isic,
        'range': lambda: validator.in_range(kwargs.get('min', float('-inf')), kwargs.get('max', float('inf'))),
        'list': lambda: validator.in_list(kwargs.get('valid_values', []))
    }
    
    if validation_type not in validation_methods:
        raise ValueError(f"Unknown validation type: {validation_type}")
    
    return validation_methods[validation_type]()


# Backward compatibility with old code
class BadValueInput:
    """Legacy validator class for backward compatibility"""
    
    def __init__(self, value):
        self.value = value
        self.validator = InputValidator(value)
    
    def boolean_type(self):
        return self.validator.is_boolean()
    
    def integer_type(self):
        return self.validator.is_integer()
    
    def string_type(self):
        return self.validator.is_string()
    
    def int_str_type(self):
        return self.validator.is_integer_string()
    
    def date_type(self):
        return self.validator.is_date()