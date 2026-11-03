"""Tokenizer turning PDDL source text into nested lists of tokens.

PDDL is an s-expression language, so we lex into parentheses-delimited nested
lists of atoms (strings). Line comments start with ``;``. Case is insensitive
in PDDL, so all tokens are lower-cased.
"""

from __future__ import annotations

from .ast import PDDLError


def tokenize(text: str) -> list:
    """Parse PDDL source into a nested list of lower-cased string tokens.

    An unbalanced parenthesis is reported with the line it is on (for a missing
    ``)``, the line of the ``(`` that was never closed).
    """
    stack: list[list] = [[]]
    opened_on: list[int] = []  # line of each '(' still open
    for number, line in enumerate(text.splitlines(), start=1):
        comment = line.find(";")
        if comment != -1:
            line = line[:comment]
        # Pad parens so a simple split yields clean tokens.
        for tok in line.replace("(", " ( ").replace(")", " ) ").split():
            if tok == "(":
                new: list = []
                stack[-1].append(new)
                stack.append(new)
                opened_on.append(number)
            elif tok == ")":
                if len(stack) == 1:
                    raise PDDLError(
                        f"Unbalanced parentheses: unexpected ')' on line {number}"
                    )
                stack.pop()
                opened_on.pop()
            else:
                stack[-1].append(tok.lower())

    if len(stack) != 1:
        raise PDDLError(
            "Unbalanced parentheses: missing ')' for the '(' opened on line "
            f"{opened_on[-1]}"
        )
    top = stack[0]
    if len(top) != 1 or not isinstance(top[0], list):
        raise PDDLError("Expected a single top-level s-expression")
    return top[0]
