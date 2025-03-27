"""Utilities for validating bioinformatics prompts."""

import re
import os
import json
from typing import List, Dict, Optional, Union
from pathlib import Path

from templates.prompt_template import BioinformaticsPrompt


def validate_prompt(prompt: BioinformaticsPrompt) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate a bioinformatics prompt for common issues.
    
    Args: prompt: BioinformaticsPrompt to validate
    Returns: Dictionary with validation results
    """
    results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check for empty or short description
    if not prompt.description or len(prompt.description) < 50:
        results["warnings"].append("Description is too short or empty")
    
    # Check for minimum number of key concepts
    if len(prompt.key_concepts) < 3:
        results["warnings"].append("Less than 3 key concepts provided")
    
    # Check for minimum number of tools
    if len(prompt.common_tools) < 3:
        results["warnings"].append("Less than 3 common tools provided")
    
    # Check for minimum number of file formats
    if len(prompt.common_file_formats) < 2:
        results["warnings"].append("Less than 2 file formats provided")
    
    # Check for minimum number of examples
    if len(prompt.examples) < 1:
        results["errors"].append("At least one example must be provided")
        results["valid"] = False
    
    # Check example quality
    for i, example in enumerate(prompt.examples):
        if len(example.query) < 10:
            results["warnings"].append(f"FewShotExample {i+1} query is very short")
        
        if len(example.context) < 20:
            results["warnings"].append(f"FewShotExample {i+1} context is very short")
        
        if len(example.response) < 100:
            results["warnings"].append(f"FewShotExample {i+1} response is very short")
        
        # Check if examples contain code blocks
        if not re.search(r'```[a-z]*\n.*?\n```', example.response, re.DOTALL):
            results["warnings"].append(f"FewShotExample {i+1} response doesn't contain code blocks")
    
    return results


def batch_validate_prompts(prompt_dir: str) -> Dict[str, Dict]:
    """
    Validate all prompt JSON files in a directory.
    
    Args: prompt_dir: Directory containing prompt JSON files
    Returns: Dictionary mapping filenames to validation results
    """
    results = {}
    prompt_path = Path(prompt_dir)
    
    for file_path in prompt_path.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                prompt = BioinformaticsPrompt.from_json(f.read())
            
            validation = validate_prompt(prompt)
            results[file_path.name] = validation
        except Exception as e:
            results[file_path.name] = {
                "valid": False,
                "errors": [f"Failed to load or validate: {str(e)}"],
                "warnings": []
            }
    
    return results


def run_test_query(prompt: BioinformaticsPrompt, test_query: str) -> str:
    """
    Run a test query through a prompt to check formatting.
    
    Args: prompt: BioinformaticsPrompt to test; test_query: Test query to run
    Returns: Formatted prompt string
    """
    return prompt.generate_prompt(test_query)


def export_all_prompts(prompts_dict: Dict[str, BioinformaticsPrompt], output_dir: str) -> None:
    """
    Export all prompts to JSON files.
    
    Args:
        prompts_dict: Dictionary mapping names to BioinformaticsPrompt objects
        output_dir: Directory to save JSON files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, prompt in prompts_dict.items():
        output_path = os.path.join(output_dir, f"{name.lower()}_prompt.json")
        
        with open(output_path, "w") as f:
            f.write(prompt.to_json())
            
    print(f"Exported {len(prompts_dict)} prompts to {output_dir}")


if __name__ == "__main__":
    # FewShotExample usage
    from prompt.prompt_genomics import genomics_prompt
    
    # Validate a single prompt
    validation_result = validate_prompt(genomics_prompt)
    print(f"Validation result: {json.dumps(validation_result, indent=2)}")
    
    # Test with a query
    test_query = "How do I analyze RNA-seq data from a non-model organism?"
    formatted_prompt = run_test_query(genomics_prompt, test_query)
    print(f"\nFormatted prompt preview:\n{formatted_prompt[:300]}...\n")
    
    # Export the prompt
    export_all_prompts({"genomics": genomics_prompt}, "exported_prompts")