import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from bioinformatics_prompts.exceptions import (
    MissingAPIKeyError,
    NoTemplateLoadedError,
    RoutingUnavailableError,
    TemplateLoadError,
    TemplateNotFoundError,
)
from bioinformatics_prompts.matching import match_area
from bioinformatics_prompts.prompt.templates.prompt_template import BioinformaticsPrompt

# NOTE: dspy is deliberately NOT imported here. It is an optional `routing`
# extra, and importing this module must not pull it in — see route_template().

# Last-resort default model, used only if a model isn't passed explicitly
# and querying the Models API for a current one fails (see
# ClaudeInteraction._resolve_default_model).
FALLBACK_MODEL = "claude-sonnet-4-6"

logger = logging.getLogger(__name__)


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
        raise MissingAPIKeyError(
            "Claude API key not provided or found in environment variables"
        )

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
        logger.warning("Skipping unreadable template %s: %s", file_path, e)
    
    return templates
  
  def load_template(self, name: str) -> BioinformaticsPrompt:
    """
    Load a prompt template by name.

    Args:
        name: The template's research_area (e.g. "Genomics") or its filename
            stem (e.g. "genomics_prompt"). Matching is exact and
            case-insensitive.

    Returns: The loaded BioinformaticsPrompt, also stored on self.prompt_template.

    Raises:
        TemplateNotFoundError: if no template has that research_area or stem.
            Matching is deliberately exact rather than reusing match_area()'s
            substring fallback: that is right for routing, where the input is
            fuzzy model output, but an explicit call should be deterministic —
            substring matching would let adding a template silently change what
            an existing call returns.
        TemplateLoadError: if the file was found but could not be read or parsed.
    """
    templates = self.list_available_templates()

    if not templates:
        raise TemplateNotFoundError(f"No prompt templates found in {self.prompt_dir}")

    wanted = name.strip().lower()
    selected = next(
        (
            template
            for template in templates
            if template["research_area"].strip().lower() == wanted
            or Path(template["filename"]).stem.lower() == wanted
        ),
        None,
    )

    if selected is None:
        available = ", ".join(sorted(t["research_area"] for t in templates))
        raise TemplateNotFoundError(
            f"No template named {name!r}. Available research areas: {available}"
        )

    return self._load_template_file(selected)

  def _load_template_file(self, template: Dict[str, str]) -> BioinformaticsPrompt:
    """Read and parse one template dict from list_available_templates()."""
    try:
      with open(template["filename"], "r") as f:
          self.prompt_template = BioinformaticsPrompt.from_json(f.read())
    except Exception as e:
      raise TemplateLoadError(
          f"Could not load template {template['research_area']!r} "
          f"from {template['filename']}: {e}"
      ) from e

    return self.prompt_template

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
        raise TemplateNotFoundError(f"No prompt templates found in {self.prompt_dir}")

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

    return matched

  def load_template_by_query(self, user_query: str) -> Optional[BioinformaticsPrompt]:
    """
    Route a user query to a template and load it.

    Args:
        user_query: The user's bioinformatics question.

    Returns: The loaded BioinformaticsPrompt, or None if routing found no
        match — callers should then pick a template explicitly with
        load_template(name).

    Raises:
        RoutingUnavailableError: if the `routing` extra is not installed.
        TemplateLoadError: if a template matched but could not be loaded.
    """
    matched = self.route_template(user_query)

    if not matched:
        return None

    return self._load_template_file(matched)

  def generate_prompt(self, user_query: str) -> str:
    """
    Generate a prompt for Claude based on a user query.
    Args: user_query: The user's bioinformatics question
    Returns: Formatted prompt string
    """
    if not self.prompt_template:
        raise NoTemplateLoadedError(
            "No prompt template loaded. Call load_template(name) or "
            "load_template_by_query(query) first."
        )
    
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
  
  def reset_conversation(self) -> None:
    """Clear the conversation history."""
    self.conversation_history = []
  
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
      logger.warning("Could not query available models (%s); using %s", e, FALLBACK_MODEL)

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

    Raises:
        anthropic.AnthropicError: propagated unchanged. Every SDK error —
            RateLimitError, AuthenticationError, APIConnectionError and the
            rest — descends from that single root, so callers can catch the
            family in one handler. Wrapping them in a package type was
            considered and rejected: it would flatten a genuinely useful
            hierarchy, and `anthropic` is already a hard dependency.
    """
    import anthropic

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

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=self.system_prompt,
        messages=messages
    )

    if use_history:
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response.content[0].text})

    return response.content[0].text

  def ask_claude(self, user_query: str, model: str = None, max_tokens: int = 4000,
                use_history: bool = True, use_template: bool = True) -> str:
    """
    Process a user query and get a response from Claude.

    Args:
        user_query: The user's bioinformatics question
        model: Claude model to use (defaults to self.default_model)
        max_tokens: Maximum tokens in response
        use_history: Whether to include conversation history
        use_template: Whether to use the loaded prompt template. If False,
                      sends the raw query without formatting.

    Returns: Claude's response as a string

    Raises:
        NoTemplateLoadedError: if use_template is True and no template has been
            loaded. This used to silently invoke the interactive picker, which
            blocks on stdin in any process without a terminal.
        anthropic.AnthropicError: propagated from send_to_claude.
    """
    model = model or self.default_model

    if use_template:
        if not self.prompt_template:
            raise NoTemplateLoadedError(
                "No prompt template loaded. Call load_template(name) or "
                "load_template_by_query(query) first, or pass use_template=False."
            )
        prompt = self.generate_prompt(user_query)
    else:
        prompt = user_query

    return self.send_to_claude(
        prompt,
        model=model,
        max_tokens=max_tokens,
        use_history=use_history
    )
