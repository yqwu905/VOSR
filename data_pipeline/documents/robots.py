"""robots.txt per RFC 9309: groups by product token, `*` and `$` patterns, longest match wins.

The standard library parser matches plain prefixes only, so `Disallow: /*.pdf$` or
`Disallow: /*?` (common on Chinese portals) would wrongly allow those URLs.
"""

import re
from urllib.parse import quote, unquote, urlsplit


def _normalize(path):
    # Compare in one canonical percent-encoding; keep reserved characters literal.
    return quote(unquote(path), safe="/?=&*$:@!,;~+%-._")


def _pattern(path):
    anchored = path.endswith("$")
    body = re.escape(_normalize(path[:-1] if anchored else path)).replace(r"\*", ".*")
    return re.compile(body + ("$" if anchored else ""))


class Robots:
    def __init__(self, text, agent):
        token = agent.split("/", 1)[0].lower()
        groups, agents, rules, delay, in_rules = [], [], [], None, False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if in_rules:  # a user-agent line after rules starts a new group
                    groups.append((agents, rules, delay))
                    agents, rules, delay, in_rules = [], [], None, False
                agents.append(value.lower())
            elif key in ("allow", "disallow"):
                in_rules = True
                if value:  # an empty Disallow allows everything
                    rules.append((key == "allow", value))
            elif key == "crawl-delay":
                in_rules = True
                try:
                    delay = float(value)
                except ValueError:
                    pass
        if agents:
            groups.append((agents, rules, delay))
        named = [g for g in groups if token in g[0]]
        chosen = named or [g for g in groups if "*" in g[0]]
        # Several matching groups are merged (RFC 9309 2.2.1).
        self.rules = [(allow, len(_normalize(path)), _pattern(path)) for g in chosen for allow, path in g[1]]
        delays = [g[2] for g in chosen if g[2] is not None]
        self.crawl_delay = max(delays) if delays else None

    def allowed(self, url):
        parts = urlsplit(url)
        target = _normalize((parts.path or "/") + ("?" + parts.query if parts.query else ""))
        if target == "/robots.txt":
            return True
        best = None  # (length, allow): longest match wins, allow wins ties
        for allow, length, pattern in self.rules:
            if pattern.match(target) and (best is None or (length, allow) > best):
                best = (length, allow)
        return True if best is None else best[1]


ALLOW_ALL = Robots("", "-")
DISALLOW_ALL = Robots("User-agent: *\nDisallow: /", "-")
