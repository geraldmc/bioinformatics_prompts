import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

from prompt.templates.prompt_template import BioinformaticsPrompt


class ClaudeInteraction:
  """Class for interacting with Claude API for bioinformatics prompts."""
  
  def __init__(self, api_key: Optional[str] = None, prompt_dir: str = "prompt", 
              model: str = "claude-3-7-sonnet-20250219"):
    """
    Initialize the Claude interaction class.
    
    Args:
        api_key: Claude API key. If None, reads from CLAUDE_API_KEY environment variable.
        prompt_dir: Directory containing prompt template JSON files.
        model: Default Claude model to use.
    """
    self.api_key = api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not self.api_key:
        raise ValueError("Claude API key not provided or found in environment variables")
    
    self.prompt_dir = prompt_dir
    self.prompt_template = None
    self.default_model = model
    
    # For managing conversation history
    self.conversation_history = []
    self.system_prompt = None
      
  def list_available_templates(self) -> List[Dict[str, str]]:
    """
    List all available prompt templates in the prompt directory,
    sorted alphabetically by filename.
    
    Returns: List of dictionaries with template information (filename, research_area)
    """
    templates = []
    
    # Find all .json files in the prompt directory and its subdirectories
    json_files = list(Path(self.prompt_dir).glob("**/*.json"))
    
    # Sort files alphabetically by stem (filename without extension and path)
    json_files.sort(key=lambda path: path.stem.lower())
    
    for idx, file_path in enumerate(json_files, 1):
      try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Get "research_area" from the JSON data if available, otherwise use "Unknown"
        research_area = data.get("research_area", "Unknown")
        
        templates.append({
            "id": idx,
            "filename": str(file_path),
            "research_area": research_area
        })
      except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {file_path}: {str(e)}")
    
    return templates
  
  def load_prompt_template(self, interactive: bool = True) -> Optional[BioinformaticsPrompt]:
    """
    Load a prompt template interactively or by filename.
    Args:
        interactive: If True, present list of templates for user to select.
                    If False, use the default template or raise error if none available.
    
    Returns: Selected BioinformaticsPrompt if successful, None otherwise
    """
    templates = self.list_available_templates()
    
    if not templates:
        print(f"No prompt templates found in {self.prompt_dir}")
        return None
    
    selected_template = None
    
    if interactive:
      # Display available templates
      print("\nAvailable research areas (prompt templates):")
      for template in templates:
          print(f"{template['id']}. {template['research_area']}")
      
      # Get user choice
      while selected_template is None:
        try:
          choice = input("\nSelect a template by number (or 'q' to quit): ")
          
          if choice.lower() == 'q':
              return None
          
          choice_idx = int(choice)
          selected_template = next((t for t in templates if t["id"] == choice_idx), None)
          
          if not selected_template:
              print(f"Invalid selection. Please choose a number between 1 and {len(templates)}")
        except ValueError:
          print("Please enter a valid number")
    else:
      # Default to first template
      selected_template = templates[0]
      print(f"Using default template: {selected_template['research_area']}")
    
    # Load the selected template
    try:
      with open(selected_template["filename"], "r") as f:
          self.prompt_template = BioinformaticsPrompt.from_json(f.read())
      print(f"Loaded template: {selected_template['research_area']}")
      return self.prompt_template
    except Exception as e:
      print(f"Error loading template: {str(e)}")
      return None
  
  def generate_prompt(self, user_query: str) -> str:
    """
    Generate a prompt for Claude based on a user query.
    Args: user_query: The user's bioinformatics question
    Returns: Formatted prompt string
    """
    if not self.prompt_template:
        raise ValueError("No prompt template loaded. Call load_prompt_template() first.")
    
    return self.prompt_template.generate_prompt(user_query)

  def set_system_prompt(self, system_prompt: Optional[str] = None) -> None:
      """
      Set or update the system prompt for conversations with Claude.
      
      Args:
          system_prompt: The system prompt to use. If None, a default bioinformatics prompt is used.
      """
      if system_prompt is None:
          # Default bioinformatics-focused system prompt
          self.system_prompt = (
              "You are Claude, an AI assistant with expertise in bioinformatics. "
              "Provide detailed, accurate responses to questions about genomics, "
              "proteomics, sequence analysis, and other bioinformatics topics. "
              "Include code examples where appropriate. Emphasize reproducibility "
              "in your code examples and explain your solutions thoroughly."
          )
      else:
          self.system_prompt = system_prompt
          
      print(f"System prompt updated: {self.system_prompt[:50]}...")
  
  def reset_conversation(self) -> None:
    """Clear the conversation history."""
    self.conversation_history = []
    print("Conversation history cleared.")
  
  def get_conversation_history(self) -> List[Dict[str, str]]:
    """Get the current conversation history."""
    return self.conversation_history.copy()
  
  def send_to_claude(self, prompt: str, model: str = None, max_tokens: int = 4000, 
                    use_history: bool = False) -> str:
    """
    Send a prompt to Claude API and get the response.
    
    Args:
        prompt: The formatted prompt to send
        model: Claude model to use (defaults to self.default_model)
        max_tokens: Maximum tokens in response (default: 4000)
        use_history: Whether to include conversation history
        
    Returns: Claude's response as a string
    """
    try:
        import anthropic
        
        # Use default model if none specified
        model = model or self.default_model
        
        # Initialize the client
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Set up system prompt if not already set
        if self.system_prompt is None:
            self.set_system_prompt()
        
        # Prepare messages
        if use_history and self.conversation_history:
            messages = self.conversation_history.copy()
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Send the request
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=messages
        )
        
        # Update conversation history if using it
        if use_history:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response.content[0].text})
        
        # Extract and return the response text
        return response.content[0].text
        
    except ImportError:
        print("Error: anthropic package not installed. Run 'pip install anthropic' to install.")
        return "Unable to communicate with Claude API due to missing dependencies."
    except Exception as e:
        print(f"Error communicating with Claude API: {str(e)}")
        return f"Error: {str(e)}"
  
  def ask_claude(self, user_query: str, model: str = None, max_tokens: int = 4000,
                show_prompt: bool = False, use_history: bool = True, 
                use_template: bool = True) -> str:
    """
    Process a user query and get a response from Claude.
    
    Args:
        user_query: The user's bioinformatics question
        model: Claude model to use (defaults to self.default_model)
        max_tokens: Maximum tokens in response
        show_prompt: Whether to print the generated prompt (useful for debugging)
        use_history: Whether to include conversation history
        use_template: Whether to use the loaded prompt template. If False,
                      sends the raw query without formatting.
        
    Returns:
        Claude's response as a string
    """
    model = model or self.default_model
    
    # Determine if we need to process the query through a template
    if use_template:
        # Make sure we have a prompt template loaded
        if not self.prompt_template:
            print("No prompt template loaded. Loading template...")
            if not self.load_prompt_template():
                return "Error: Failed to load a prompt template."
        
        # Generate the formatted prompt using the template
        try:
            prompt = self.generate_prompt(user_query)
        except Exception as e:
            print(f"Error generating prompt: {str(e)}")
            return f"Error generating prompt: {str(e)}"
    else:
        # Use the raw query without template formatting
        prompt = user_query
    
    # Optionally show the prompt for debugging
    if show_prompt:
        print("\n===== PROMPT SENT TO CLAUDE =====")
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        print("=================================\n")
    
    # Send to Claude and get response
    try:
        return self.send_to_claude(
            prompt, 
            model=model, 
            max_tokens=max_tokens,
            use_history=use_history
        )
    except Exception as e:
        print(f"Error in ask_claude: {str(e)}")
        return f"An error occurred: {str(e)}"


  def start_conversation(self, use_template: bool = True) -> None:
    """
    Start an interactive conversation with Claude in the terminal.
    
    Args:
        use_template: Whether to format queries with the loaded template
    """
    # Start with a clean conversation history
    self.reset_conversation()
    
    # Load prompt template if using templates
    if use_template and not self.prompt_template:
        if not self.load_prompt_template(interactive=True):
            print("Failed to load template. Exiting conversation.")
            return
            
    print("\n=== Starting conversation with Claude ===")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'reset' to clear the conversation history")
    print("Type 'template' to load a different template")
    print("================================================\n")
    
    while True:
        # Get user input
        user_query = input("\nYou: ")
        
        # Check for exit commands
        if user_query.lower() in ('quit', 'exit', 'bye'):
            print("Ending conversation. Goodbye!")
            break
            
        # Check for special commands
        if user_query.lower() == 'reset':
            self.reset_conversation()
            print("Conversation history has been reset.")
            continue
            
        if user_query.lower() == 'template':
            self.load_prompt_template(interactive=True)
            continue
        
        # Get response from Claude
        response = self.ask_claude(
            user_query, 
            use_template=use_template, 
            use_history=True
        )
        
        # Print Claude's response
        print("\nClaude:", response)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Try to load API key from .env file
    load_dotenv()
    
    # Initialize the interaction class
    try:
        interaction = ClaudeInteraction(prompt_dir="prompt")
        
        # Start an interactive conversation
        interaction.start_conversation()
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Please set your ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable.")
        print("You can create a .env file with: ANTHROPIC_API_KEY=your_key_here")