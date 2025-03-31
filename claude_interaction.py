import anthropic
import json
from prompt.templates.prompt_template import BioinformaticsPrompt

class ClaudeInteraction:
    """Manages interactions with Claude using bioinformatics prompt templates."""
    
    def __init__(self, api_key, model="claude-3-7-sonnet-20250219"):
        """
        Initialize the Claude interaction client.
        
        Args:
            api_key (str): Anthropic API key
            model (str): Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.conversation_messages = []
        self.system_prompt = None
    
    def load_prompt_template(self, template_path):
        """
        Load a bioinformatics prompt template from a JSON file.
        
        Args:
            template_path (str): Path to the template JSON file
            
        Returns:
            BioinformaticsPrompt: Loaded prompt template
        """
        with open(template_path, 'r') as f:
            template_json = f.read()
        
        return BioinformaticsPrompt.from_json(template_json)
    
    def create_system_prompt(self, prompt_template):
        """
        Create a system prompt from the bioinformatics template.
        
        Args:
            prompt_template (BioinformaticsPrompt): The template to use
            
        Returns:
            str: Formatted system prompt
        """
        # Create a simplified version of the template for the system prompt
        system_prompt = [
            f"# {prompt_template.research_area} Research Context\n",
            f"{prompt_template.description}\n",
            "## Key Concepts\n"
        ]
        
        for concept in prompt_template.key_concepts:
            system_prompt.append(f"- {concept}\n")
        
        system_prompt.append("\n## Common Tools\n")
        for tool in prompt_template.common_tools:
            system_prompt.append(f"- {tool}\n")
            
        system_prompt.append("\n## Common File Formats\n")
        for fmt in prompt_template.common_file_formats:
            system_prompt.append(f"- {fmt['name']}: {fmt['description']}\n")
        
        # Include a few examples to demonstrate expected response pattern
        system_prompt.append("\n## Examples of how to respond to user queries:\n")
        for i, example in enumerate(prompt_template.examples, 1):
            system_prompt.append(f"### Example {i}\n")
            system_prompt.append(f"User Query: {example.query}\n")
            system_prompt.append(f"Context: {example.context}\n")
            system_prompt.append(f"You should respond like this: {example.response}\n\n")
            
        system_prompt.append("\nPlease answer the user's questions about bioinformatics with detailed, accurate information and reproducible code examples where appropriate.")
        
        return "".join(system_prompt)
    
    def ask_question(self, question, prompt_template=None, max_tokens=1000):
        """
        Ask Claude a question using the bioinformatics prompt template.
        
        Args:
            question (str): The user's question
            prompt_template (BioinformaticsPrompt, optional): Prompt template to use
            max_tokens (int): Maximum number of tokens in the response
            
        Returns:
            str: Claude's response
        """
        # Set or update system prompt if a template is provided
        if prompt_template:
            self.system_prompt = self.create_system_prompt(prompt_template)
            # Reset conversation if switching templates
            self.conversation_messages = []
        
        # Add the user's question to conversation history
        self.conversation_messages.append({"role": "user", "content": question})
        
        # Create the request to Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,  # system is a top-level parameter
            messages=self.conversation_messages
        )
        
        # Add Claude's response to conversation history
        self.conversation_messages.append({"role": "assistant", "content": response.content[0].text})
        
        return response.content[0].text
    
    def reset_conversation(self):
        """Reset the conversation history but keep the system prompt."""
        self.conversation_messages = []


# Example usage
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path
    
    from dotenv import load_dotenv

    load_dotenv()  # take environment variables

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Please enter your Anthropic API key: ")
        os.environ["ANTHROPIC_API_KEY"] = api_key

    # Initialize the Claude interaction
    claude = ClaudeInteraction(api_key)
    
    # Load a genomics prompt template
    genomics_template = claude.load_prompt_template("prompt/genomics_prompt.json")
    
    # Ask a question using the genomics template
    question = "How do I assemble a bacterial genome from Illumina paired-end reads?"
    response = claude.ask_question(question, genomics_template)
    
    print(f"Question: {question}")
    print(f"Response: {response}")
    
    # Continue the conversation
    follow_up = "What quality control steps should I include?"
    response = claude.ask_question(follow_up)
    
    print(f"Follow-up: {follow_up}")
    print(f"Response: {response}")
    
    # Reset the conversation to start fresh with a new template
    claude.reset_conversation()