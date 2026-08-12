from typing import List, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class FewShotExample:
    """A few-shot example for the prompt."""
    query: str
    context: str
    response: str
    
    def format(self) -> str:
        """Format the example as a string."""
        return f"User Query: {self.query}\nContext: {self.context}\nResponse: {self.response}\n"


class BioinformaticsPrompt:
    """Base class for bioinformatics research area prompts."""
    
    def __init__(
        self,
        research_area: str,
        description: str,
        key_concepts: List[str],
        common_tools: List[str],
        common_file_formats: List[Dict[str, str]],
        examples: List[FewShotExample],
        references: Optional[List[str]] = None
    ):
        """
        Initialize a bioinformatics prompt template.
        
        Args:
            research_area: Name of the research area
            description: Brief description of the research area
            key_concepts: List of key concepts in this area
            common_tools: List of common tools used in this area
            common_file_formats: List of common file formats with descriptions
            examples: List of few-shot examples
            references: Optional list of references
        """
        self.research_area = research_area
        self.description = description
        self.key_concepts = key_concepts
        self.common_tools = common_tools
        self.common_file_formats = common_file_formats
        self.examples = examples
        self.references = references or []
        
    def generate_prompt(self, user_query: str) -> str:
        """
        Generate a complete prompt based on the user query.
        
        Args:
            user_query: The user's query
            
        Returns:
            A formatted prompt string
        """
        prompt = [
            f"# {self.research_area} Research Context\n",
            f"{self.description}\n",
            "## Key Concepts\n"
        ]
        
        for concept in self.key_concepts:
            prompt.append(f"- {concept}\n")
        
        prompt.append("\n## Common Tools\n")
        for tool in self.common_tools:
            prompt.append(f"- {tool}\n")
            
        prompt.append("\n## Common File Formats\n")
        for fmt in self.common_file_formats:
            prompt.append(f"- {fmt['name']}: {fmt['description']}\n")
        
        prompt.append("\n## FewShotExamples\n")
        for i, example in enumerate(self.examples, 1):
            prompt.append(f"### FewShotExample {i}\n{example.format()}\n")
            
        if self.references:
            prompt.append("\n## References\n")
            for ref in self.references:
                prompt.append(f"- {ref}\n")
                
        prompt.append(f"\n## Current Query\n{user_query}")
        
        return "".join(prompt)
    
    def to_json(self) -> str:
        """
        Convert the prompt template to a JSON string.
        Returns: JSON representation of the prompt template
        """
        return json.dumps({
            "research_area": self.research_area,
            "description": self.description,
            "key_concepts": self.key_concepts,
            "common_tools": self.common_tools,
            "common_file_formats": self.common_file_formats,
            "examples": [
                {
                    "query": ex.query,
                    "context": ex.context,
                    "response": ex.response
                } for ex in self.examples
            ],
            "references": self.references
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BioinformaticsPrompt':
        """
        Create a prompt template from a JSON string.
        
        Args: json_str: JSON representation of the prompt template
        Returns: A BioinformaticsPrompt instance
        """
        data = json.loads(json_str)
        examples = [
            FewShotExample(
                query=ex["query"],
                context=ex["context"],
                response=ex["response"]
            ) for ex in data["examples"]
        ]
        
        return cls(
            research_area=data["research_area"],
            description=data["description"],
            key_concepts=data["key_concepts"],
            common_tools=data["common_tools"],
            common_file_formats=data["common_file_formats"],
            examples=examples,
            references=data.get("references", [])
        )