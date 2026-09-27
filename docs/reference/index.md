# API reference

Everything on these pages is named in `bioinformatics_prompts.__all__` and is
importable directly from the package:

```python
from bioinformatics_prompts import ClaudeInteraction, BioinformaticsPrompt
```

Twelve names, grouped by what you reach for them to do:

| Page | What it covers |
|---|---|
| [Client](client.md) | `ClaudeInteraction` — load a template, ask Claude a question. `TemplateInfo` describes an available template. |
| [Templates](templates.md) | `BioinformaticsPrompt` and `FewShotExample` — the data model. `validate_prompt` and `ValidationResult` — check one before you rely on it. |
| [Exceptions](exceptions.md) | The six-member hierarchy every failure raises from. |

!!! note "What is not here"

    Anything outside `__all__` — `cli_chat`, `matching`, `dspy_modules`, and
    the rest of `utils.validation` — is internal and may move without notice.
    It is importable, but not promised.

The pages are generated from the docstrings in the source, so they cannot drift
from the code they describe. A test asserts they cover exactly `__all__`.
