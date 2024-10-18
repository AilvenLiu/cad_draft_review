from utils.rule_parser import parse_rules_from_excel, parse_rules_from_text
from utils.knowledge_graph import KnowledgeGraph

def initialize_knowledge_base(rules_source, source_type='excel', uri='bolt://localhost:7687', user='neo4j', password='password'):
    """
    Initializes the knowledge base by parsing rules and populating the knowledge graph.

    Args:
        rules_source (str): Path to the rules file.
        source_type (str): Type of the rules file ('excel' or 'text').
        uri (str): Neo4j URI.
        user (str): Neo4j username.
        password (str): Neo4j password.
    """
    if source_type == 'excel':
        rules = parse_rules_from_excel(rules_source)
    elif source_type == 'text':
        rules = parse_rules_from_text(rules_source)
    else:
        raise ValueError("Unsupported source type. Use 'excel' or 'text'.")
    
    kg = KnowledgeGraph(uri, user, password)
    for rule in rules:
        kg.add_rule(rule)
    kg.close()

def validate_detections(detections, uri='bolt://localhost:7687', user='neo4j', password='password'):
    """
    Validates detections against the knowledge base rules.

    Args:
        detections (List[dict]): List of detected signs with codes and connections.
        uri (str): Neo4j URI.
        user (str): Neo4j username.
        password (str): Neo4j password.

    Returns:
        List[dict]: List of rule violations with suggestions.
    """
    kg = KnowledgeGraph(uri, user, password)
    errors = []
    for sign in detections:
        sign_type = sign['type']
        sign_code = sign.get('code', None)
        rules = kg.get_rules_for_sign(sign_type)
        for rule in rules:
            condition = rule['condition']
            action = rule['action']
            # Implement condition checking logic based on sign_type and sign_code
            # This requires parsing the condition string and applying it
            # Example condition: "Protector must follow Pump with code P-20058"
            if not check_condition(sign, detections, condition):
                errors.append({
                    'sign': sign,
                    'rule_violated': rule['description'],
                    'suggestion': action
                })
    kg.close()
    return errors

def check_condition(sign, detections, condition):
    """
    Checks if a condition is satisfied based on the detections.

    Args:
        sign (dict): Detected sign.
        detections (List[dict]): All detections.
        condition (str): Condition string from the rule.

    Returns:
        bool: True if condition is satisfied, False otherwise.
    """
    # Implement condition parsing and checking logic
    # This is simplified and should be expanded based on actual rule formats
    if "Protector" in condition and "Pump" in condition:
        pump_code = re.search(r'Pump with code (\w+)', condition)
        protector_required = True
        if pump_code:
            pump_code = pump_code.group(1)
            # Check if Protector exists after Pump with specified code
            for det in detections:
                if det['type'] == 'Pump' and det.get('code') == pump_code:
                    pump_index = detections.index(det)
                    # Check signs after Pump
                    if pump_index + 1 < len(detections):
                        next_det = detections[pump_index + 1]
                        if next_det['type'] != 'Protector':
                            return False
        return True
    return True