"""Write the template catalogue into the docs build (#24).

Run by mkdocs-gen-files on every build, from `mkdocs.yml`. The rendering lives
in `template_catalogue.py` beside this file, so the test suite can exercise it
without installing mkdocs -- the core-only CI job runs the tests with no `docs`
dependency group.

In `scripts/` rather than `docs/` deliberately: anything inside `docs_dir` that
is not Markdown is copied into the built site as a static asset, so a generator
living there would publish its own source alongside the pages it writes.

The output directory is `catalogue/`, **not** `templates/`, and that is not a
style choice. MkDocs excludes `/templates/` from every build by default --
`mkdocs.structure.files._default_exclude` is `['.*', '/templates/']` -- because
themes keep their Jinja templates there. A catalogue written to `templates/`
is generated, appears in the nav, and is then silently dropped from the site,
with `mkdocs build --strict` still exiting 0.

Nothing here touches the working tree. `mkdocs_gen_files.open()` writes into the
build's virtual file tree, so `docs/templates/` never exists on disk and there
is no generated Markdown to commit or keep in step.
"""

import sys
from pathlib import Path

import mkdocs_gen_files

# gen-files runs this through runpy.run_path, which does not put the script's
# own directory on sys.path the way `python script.py` would.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from template_catalogue import (  # noqa: E402
    iter_templates,
    render_index,
    render_page,
    render_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = REPO_ROOT / "src" / "bioinformatics_prompts" / "prompt"

templates = list(iter_templates(PROMPT_DIR))
if not templates:
    raise SystemExit(f"no template JSON found in {PROMPT_DIR}")

with mkdocs_gen_files.open("catalogue/index.md", "w") as page:
    page.write(render_index(templates))

for path, data in templates:
    with mkdocs_gen_files.open(f"catalogue/{path.stem}.md", "w") as page:
        page.write(render_page(data))

# literate-nav reads this to build the Templates section; mkdocs.yml delegates
# to it with the trailing slash in `Templates: catalogue/`.
#
# Named `.nav.md` rather than the default `SUMMARY.md` so it does not ship as a
# page of its own: literate-nav marks its nav file NOT_IN_NAV, which still gets
# built, whereas a leading dot matches MkDocs' `.*` default exclusion and drops
# it from the site entirely.
with mkdocs_gen_files.open("catalogue/.nav.md", "w") as summary:
    summary.write(render_summary(templates))
