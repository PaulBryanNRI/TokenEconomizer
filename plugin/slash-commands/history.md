---
description: Show which earlier Claude Code sessions TokenEconomizer imported for this repository, or import the ones still waiting
argument-hint: "[show | run]"
---
The arguments I typed after the command are: "$ARGUMENTS" (it shows the import when there are none).
Run this exact command with the Bash tool, then show me its output verbatim in a code block:

```
te="${CLAUDE_PLUGIN_ROOT}/scripts/te"; [ -x "$te" ] || te="${TOKENECONOMIZER_HOME:-$HOME/.tokeneconomizer}/bin/te"; "$te" --cwd "$(pwd)" history $ARGUMENTS
```

It imports sessions only when I typed `run` in those arguments; never add it yourself.
