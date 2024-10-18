import re
import pandas as pd

def parse_rules_from_excel(excel_path: str) -> list:
    """
    Parses rules from an Excel file into structured dictionaries.

    Args:
        excel_path (str): Path to the Excel file containing rules.

    Returns:
        List[dict]: List of rules with conditions and actions.
    """
    df = pd.read_excel(excel_path)
    rules = []
    for _, row in df.iterrows():
        rule = {
            'description': row['Description'],
            'condition': row['Condition'],
            'action': row['Action']
        }
        rules.append(rule)
    return rules

def parse_rules_from_text(text_path: str) -> list:
    """
    Parses rules from a text file into structured dictionaries.

    Args:
        text_path (str): Path to the text file containing rules.

    Returns:
        List[dict]: List of rules with conditions and actions.
    """
    with open(text_path, 'r') as file:
        lines = file.readlines()
    
    rules = []
    for line in lines:
        # Simple parsing assuming format: "Condition -> Action"
        parts = line.strip().split('->')
        if len(parts) == 2:
            rule = {
                'description': line.strip(),
                'condition': parts[0].strip(),
                'action': parts[1].strip()
            }
            rules.append(rule)
    return rules