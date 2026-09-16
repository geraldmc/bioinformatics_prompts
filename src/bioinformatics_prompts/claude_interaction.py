import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from bioinformatics_prompts.exceptions import RoutingUnavailableError
from bioinformatics_prompts.matching import match_area
from bioinformatics_prompts.prompt.templates.prompt_template import BioinformaticsPrompt

# NOTE: dspy is deliberately NOT imported here. It is an optional `routing`
# extra, and importing this module must not pull it in — see route_template().

# Last-resort default model, used only if a model isn't passed explicitly
# and querying the Models API for a current one fails (see
# ClaudeInteraction._resolve_default_model).
FALLBACK_MODEL = "claude-sonnet-4-6"


class ClaudeInteraction:
  """Class for interacting with Claude API for bioinformatics prompts."""

  def __init__(self, api_key: Optional[str] = None, prompt_dir: Optional[str] = None,
              model: Optional[str] = None, require_api_key: bool = True):
    """
    Initialize the Claude interaction class.

    Args:
        api_key: Claude API key. If None, reads from CLAUDE_API_KEY environment variable.
        prompt_dir: Directory containing prompt template JSON files. If None,
            defaults to the `prompt` directory shipped alongside this module,
            resolved independently of the current working directory.
        model: Default Claude model to use. If None, a default is resolved
            lazily on first use (see _resolve_default_model) rather than
            at construction time, so instantiating this class never
            requires network access.
        require_api_key: If False, skip raising when no API key is found,
            leaving self.api_key as None. For callers that only need
            functionality that doesn't touch the Claude API (e.g. listing
            templates).
    """
    self.api_key = api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if require_api_key and not self.api_key:
        raise ValueError("Claude API key not provided or found in environment variables")

    self.prompt_dir = prompt_dir or str(Path(__file__).resolve().parent / "prompt")
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
            "research_area": research_area,
            "description": data.get("description", "")
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

  def route_template(self, user_query: str) -> Optional[Dict[str, str]]:
    """
    Use a DSPy-based router to pick the best-matching template for a
    user query, without presenting the interactive numbered menu.

    Args:
        user_query: The user's bioinformatics question.

    Returns: The matched template dict (same shape as list_available_templates()
        entries) if a match is found, None otherwise.

    Raises:
        RoutingUnavailableError: if the optional `routing` extra is not
            installed. This is deliberately not folded into the None return,
            which already means "no template matched" — a configuration error
            and a routing miss are different outcomes.
    """
    templates = self.list_available_templates()

    if not templates:
        print(f"No prompt templates found in {self.prompt_dir}")
        return None

    # Deferred: these reach dspy, which ships only with the `routing` extra.
    # Both dspy_modules.lm and dspy_modules.router touch dspy at module scope
    # (router.py subclasses dspy.Signature), so neither can be imported above.
    try:
        from bioinformatics_prompts.dspy_modules.lm import configure_claude_lm
        from bioinformatics_prompts.dspy_modules.router import TemplateRouter
    except ImportError as e:
        raise RoutingUnavailableError() from e

    model = self.default_model or FALLBACK_MODEL
    configure_claude_lm(model=model, api_key=self.api_key)

    router = TemplateRouter()
    prediction = router(question=user_query, areas=templates)
    matched = match_area(prediction.research_area, templates)

    if not matched:
        print(f"No matching template found for query: {user_query}")
        return None

    return matched

  def load_prompt_template_by_query(self, user_query: str) -> Optional[BioinformaticsPrompt]:
    """
    Route a user query to a template and load it, as an alternative to the
    interactive numbered menu in load_prompt_template.

    Args:
        user_query: The user's bioinformatics question.

    Returns: The loaded BioinformaticsPrompt if a match was found and loaded
        successfully, None otherwise (callers should fall back to the
        interactive picker, e.g. load_prompt_template()).
    """
    matched = self.route_template(user_query)

    if not matched:
        return None

    try:
      with open(matched["filename"], "r") as f:
          self.prompt_template = BioinformaticsPrompt.from_json(f.read())
      print(f"Loaded template: {matched['research_area']}")
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

  def _resolve_default_model(self, client) -> str:
    """
    Resolve a sensible default model by querying the Models API.

    Picks the most recently released model whose id contains "sonnet"
    (the API lists models most-recent-first, so the first match is the
    latest one). Falls back to FALLBACK_MODEL if the query fails or no
    matching model is found.

    Args:
        client: An initialized anthropic.Anthropic client.

    Returns: A model id string.
    """
    try:
      for model in client.models.list():
        if "sonnet" in model.id:
          return model.id
    except Exception as e:
      print(f"Error querying available models: {str(e)}")

    return FALLBACK_MODEL

  def send_to_claude(self, prompt: str, model: str = None, max_tokens: int = 4000,
                    use_history: bool = False) -> str:
    """
    Send a prompt to Claude API and get the response.

    Args:
        prompt: The formatted prompt to send
        model: Claude model to use (defaults to self.default_model, resolving
            and caching it lazily via _resolve_default_model if not yet set)
        max_tokens: Maximum tokens in response (default: 4000)
        use_history: Whether to include conversation history

    Returns: Claude's response as a string
    """
    try:
        import anthropic

        # Initialize the client
        client = anthropic.Anthropic(api_key=self.api_key)

        # Use default model if none specified, resolving and caching it lazily
        if model is None:
            if self.default_model is None:
                self.default_model = self._resolve_default_model(client)
            model = self.default_model

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
