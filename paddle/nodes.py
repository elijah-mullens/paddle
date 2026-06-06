from __future__ import annotations

from pathlib import Path


DEFAULT_NODELIST = Path("~/.config/paddle/nodelist").expanduser()


def read_nodelist(path: Path) -> list[str]:
    try:
        lines = path.expanduser().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read nodelist {path.expanduser()}: {exc}") from exc

    hosts: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        host = raw_line.split("#", 1)[0].strip()
        if not host:
            continue
        if any(character.isspace() for character in host):
            raise ValueError(
                f"invalid nodelist entry on line {line_number}: expected one SSH alias"
            )
        if host not in seen:
            hosts.append(host)
            seen.add(host)
    if not hosts:
        raise ValueError(f"nodelist {path.expanduser()} contains no hosts")
    return hosts
