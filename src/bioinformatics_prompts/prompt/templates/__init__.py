"""The authored source of the 14 bundled templates -- content, not code.

Each module declares exactly one BioinformaticsPrompt, and the JSON file it
produces is named after that variable. Nothing imports these at runtime;
they exist so a template can be edited as readable Python rather than as a
single-line JSON string (#25).
"""
