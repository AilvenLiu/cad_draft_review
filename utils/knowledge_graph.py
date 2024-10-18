from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def add_sign(self, sign):
        with self.driver.session() as session:
            session.write_transaction(self._create_sign, sign)
    
    @staticmethod
    def _create_sign(tx, sign):
        tx.run(
            "MERGE (s:Sign {code: $code, type: $type})",
            code=sign.get('code'),
            type=sign.get('type')
        )
    
    def add_rule(self, rule):
        with self.driver.session() as session:
            session.write_transaction(self._create_rule, rule)
    
    @staticmethod
    def _create_rule(tx, rule):
        tx.run(
            "CREATE (r:Rule {description: $desc, condition: $cond, action: $act})",
            desc=rule['description'],
            cond=rule['condition'],
            act=rule['action']
        )
    
    def get_rules_for_sign(self, sign_type):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_rules, sign_type)
            return result
    
    @staticmethod
    def _get_rules(tx, sign_type):
        result = tx.run(
            "MATCH (r:Rule) WHERE r.condition CONTAINS $sign_type RETURN r.description, r.condition, r.action",
            sign_type=sign_type
        )
        return [record for record in result]