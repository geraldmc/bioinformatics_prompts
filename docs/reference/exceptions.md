# Exceptions

Every failure this package raises inherits from `BioinformaticsPromptsError`, so
one `except` clause catches all of them. The library raises rather than printing
and returning `None`; see [Errors](../guide/errors.md) for how to handle each.

::: bioinformatics_prompts.exceptions.BioinformaticsPromptsError

::: bioinformatics_prompts.exceptions.MissingAPIKeyError

::: bioinformatics_prompts.exceptions.NoTemplateLoadedError

::: bioinformatics_prompts.exceptions.RoutingUnavailableError

::: bioinformatics_prompts.exceptions.TemplateLoadError

::: bioinformatics_prompts.exceptions.TemplateNotFoundError
