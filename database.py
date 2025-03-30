import hashlib
import json
import os
from datetime import datetime

"""
Database handler module for storing and managing hackathon problems.
Uses JSON file-based storage with MD5 hashing for deduplication.
Provides CRUD operations and activity logging functionality.
"""

class DatabaseHandler:
    """Manages persistent storage of hackathon problems and activity logs.
    
    Implements a simple JSON-based database with problem deduplication using MD5 hashing.
    Stores both problem data and operation logs with timestamps.
    
    Attributes:
        db_path (str): Path to the JSON database file
        data (dict): In-memory cache of database content with 'problems' and 'logs' collections
    """
    
    def __init__(self, db_path="problems_db.json"):
        """Initialize database handler with specified file path.
        
        Args:
            db_path (str): Path to JSON database file. Defaults to 'problems_db.json'
        """
        self.db_path = db_path
        self.data = self._load_data()

    def _load_data(self):
        """Load database content from JSON file.
        
        Returns:
            dict: Database content with 'problems' and 'logs' collections.
                 Returns empty structure if file doesn't exist or is corrupt.
        """
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except:
                return {"problems": [], "logs": []}
        return {"problems": [], "logs": []}

    def _save_data(self):
        """Persist current database state to JSON file.
        Handles serialization of all database content including timestamps.
        """
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add_problem(self, problem_data):
        """Add new problem to database with duplicate detection.
        
        Args:
            problem_data (dict): Problem data containing:
                - title (str): Problem title
                - description (str): Problem description
                - theme (str): Problem category/theme
                - difficulty (str): Problem difficulty level
                - source_url (str): Source of the problem
                
        Returns:
            bool: True if problem was added successfully, False if duplicate or error
        """
        problem_hash = self._generate_hash(problem_data)
        if not self._problem_exists(problem_hash):
            try:
                problem = {
                    "id": len(self.data["problems"]) + 1,
                    "title": problem_data["title"],
                    "description": problem_data["description"],
                    "theme": problem_data["theme"],
                    "difficulty": problem_data["difficulty"],
                    "source_url": problem_data["source_url"],
                    "created_at": datetime.now().isoformat(),
                    "hash": problem_hash
                }
                self.data["problems"].append(problem)
                self._save_data()
                self.log_action("add_problem", f"Added new problem: {problem_data['title']}")
                return True
            except Exception as e:
                self.log_action("error", f"Database error: {str(e)}")
                return False
        return False

    def _generate_hash(self, problem_data):
        """Generate unique hash for problem deduplication.
        
        Creates MD5 hash of problem title and description to identify duplicates.
        
        Args:
            problem_data (dict): Problem data to hash
            
        Returns:
            str: MD5 hex digest of problem content
        """
        content = f"{problem_data['title']}{problem_data['description']}"
        return hashlib.md5(content.encode()).hexdigest()

    def _problem_exists(self, problem_hash):
        """Check if problem with given hash already exists.
        
        Args:
            problem_hash (str): MD5 hash to check
            
        Returns:
            bool: True if problem exists, False otherwise
        """
        return any(p["hash"] == problem_hash for p in self.data["problems"])

    def log_action(self, action, details):
        """Record database operation in activity log.
        
        Args:
            action (str): Type of operation performed
            details (str): Description of the operation
        """
        log = {
            "id": len(self.data["logs"]) + 1,
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.data["logs"].append(log)
        self._save_data()
