from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    name: str | None = None
    length: float = 0.0
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaf_names(self) -> list[str]:
        names: list[str] = []

        def _walk(node: TreeNode) -> None:
            if node.is_leaf:
                if not node.name:
                    raise ValueError("All leaf nodes must have names.")
                names.append(node.name)
                return
            for child in node.children:
                _walk(child)

        _walk(self)
        return names

    def branch_lengths(self) -> list[float]:
        lengths: list[float] = []

        def _walk(node: TreeNode) -> None:
            for child in node.children:
                lengths.append(child.length)
                _walk(child)

        _walk(self)
        return lengths


def parse_newick(newick: str) -> TreeNode:
    text = newick.strip()
    if not text:
        raise ValueError("Newick string is empty.")
    if text.endswith(";"):
        text = text[:-1]
    if not text:
        raise ValueError("Newick string is invalid.")

    idx = 0

    def _skip_ws(pos: int) -> int:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    def _read_name(pos: int) -> tuple[str | None, int]:
        pos = _skip_ws(pos)
        start = pos
        while pos < len(text) and text[pos] not in ",():;":
            pos += 1
        token = text[start:pos].strip()
        return (token if token else None), pos

    def _read_length(pos: int) -> tuple[float, int]:
        pos = _skip_ws(pos)
        if pos >= len(text) or text[pos] != ":":
            return 0.0, pos
        pos += 1
        pos = _skip_ws(pos)
        start = pos
        while pos < len(text) and text[pos] not in ",()":
            pos += 1
        raw = text[start:pos].strip()
        if not raw:
            raise ValueError("Missing branch length after ':'.")
        try:
            value = float(raw)
        except ValueError as exc:  # pragma: no cover - defensive parse guard
            raise ValueError(f"Invalid branch length: {raw}") from exc
        if value < 0:
            raise ValueError("Branch lengths must be non-negative.")
        return value, pos

    def _parse_subtree(pos: int) -> tuple[TreeNode, int]:
        pos = _skip_ws(pos)
        if pos >= len(text):
            raise ValueError("Unexpected end of Newick string.")

        if text[pos] == "(":
            pos += 1
            children: list[TreeNode] = []
            while True:
                child, pos = _parse_subtree(pos)
                children.append(child)
                pos = _skip_ws(pos)
                if pos >= len(text):
                    raise ValueError("Unterminated internal node in Newick string.")
                if text[pos] == ",":
                    pos += 1
                    continue
                if text[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"Unexpected token '{text[pos]}' in Newick string.")

            name, pos = _read_name(pos)
            length, pos = _read_length(pos)
            return TreeNode(name=name, length=length, children=children), pos

        name, pos = _read_name(pos)
        if not name:
            raise ValueError("Leaf node is missing a name.")
        length, pos = _read_length(pos)
        return TreeNode(name=name, length=length, children=[]), pos

    root, idx = _parse_subtree(idx)
    idx = _skip_ws(idx)
    if idx != len(text):
        raise ValueError(f"Unexpected trailing content in Newick: {text[idx:]}")

    if len(set(root.leaf_names())) != len(root.leaf_names()):
        raise ValueError("Leaf names in Newick tree must be unique.")
    return root
