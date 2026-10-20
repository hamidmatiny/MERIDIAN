"""Unified diff for the remediation module the patch adds."""


def unified_diff(path: str, source: str) -> str:
    """A new-file diff whose added lines are ``source``."""
    lines = source.splitlines() or [""]
    added = "\n".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{added}\n"
    )


def source_from_diff(diff: str) -> str:
    """Recover the added file body. Lines that are not additions are ignored."""
    body: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            body.append(line[1:])
    return "\n".join(body) + ("\n" if body else "")
