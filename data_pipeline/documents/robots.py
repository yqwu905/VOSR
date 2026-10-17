"""robots.txt per RFC 9309: groups by product token, `*` and `$` patterns, longest match wins.

The standard library parser matches plain prefixes only, so `Disallow: /*.pdf$` or
`Disallow: /*?` (common on Chinese portals) would wrongly allow those URLs. Patterns are matched
without regexes or backtracking (the pieces between `*` are found leftmost, in order), so a rule
with many wildcards costs linear time. A leading UTF-8 byte order mark is ignored.

The collected documents train a model, so a site's opt-out of AI crawlers also binds us
even though our own token is different: a path disallowed for any well-known AI training
agent (GPTBot, CCBot, ClaudeBot, ...) is treated as disallowed, and a Content-Signal of
`ai-train=no` disallows everything.
"""

import re
from urllib.parse import quote, unquote, urlsplit

AI_AGENTS = {"gptbot", "ccbot", "google-extended", "anthropic-ai", "claudebot", "claude-web", "cohere-ai",
             "cohere-training-data-crawler", "bytespider", "applebot-extended", "meta-externalagent",
             "img2dataset", "ai2bot", "ai2bot-dolma", "omgilibot", "diffbot", "perplexitybot", "amazonbot"}


def _normalize(path):
    # Compare in one canonical percent-encoding; keep reserved characters literal.
    return quote(unquote(path), safe="/?=&*$:@!,;~+%-._")


def _pattern(path):
    anchored = path.endswith("$")
    return _normalize(path[:-1] if anchored else path).split("*"), anchored


def _match(pattern, target):
    # Leftmost search for each piece is enough for `*` globs: a match exists iff this one succeeds.
    pieces, anchored = pattern
    if not target.startswith(pieces[0]):
        return False
    if len(pieces) == 1:
        return not anchored or target == pieces[0]
    position = len(pieces[0])
    for piece in pieces[1:-1]:
        position = target.find(piece, position)
        if position < 0:
            return False
        position += len(piece)
    if anchored:
        return target.endswith(pieces[-1]) and len(target) - len(pieces[-1]) >= position
    return target.find(pieces[-1], position) >= 0


def _compile(groups):
    return [(allow, len(_normalize(path)), _pattern(path)) for g in groups for allow, path in g[1]]


def _verdict(rules, target):
    best = None  # (length, allow): longest match wins, allow wins ties
    for allow, length, pattern in rules:
        if _match(pattern, target) and (best is None or (length, allow) > best):
            best = (length, allow)
    return True if best is None else best[1]


class Robots:
    def __init__(self, text, agent):
        token = agent.split("/", 1)[0].lower()
        groups, agents, rules, delay, in_rules = [], [], [], None, False
        self.ai_train_no = False
        for raw in text.removeprefix(chr(0xFEFF)).splitlines():
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
            elif key == "content-signal" and re.search(r"ai-train\s*=\s*no", value, re.I):
                self.ai_train_no = True
        if agents:
            groups.append((agents, rules, delay))
        named = [g for g in groups if token in g[0]]
        chosen = named or [g for g in groups if "*" in g[0]]
        # Several matching groups are merged (RFC 9309 2.2.1).
        self.rules = _compile(chosen)
        self.ai_rules = [_compile([g]) for g in groups if AI_AGENTS & set(g[0])]
        delays = [g[2] for g in chosen if g[2] is not None]
        self.crawl_delay = max(delays) if delays else None

    def allowed(self, url):
        parts = urlsplit(url)
        target = _normalize((parts.path or "/") + ("?" + parts.query if parts.query else ""))
        if target == "/robots.txt":
            return True
        if self.ai_train_no:
            return False
        return _verdict(self.rules, target) and all(_verdict(rules, target) for rules in self.ai_rules)

    def reason(self, url):
        """Why a URL is refused: 'ai_optout' when only the AI-agent rules forbid it."""
        if self.allowed(url):
            return None
        parts = urlsplit(url)
        target = _normalize((parts.path or "/") + ("?" + parts.query if parts.query else ""))
        return "robots_disallow" if not _verdict(self.rules, target) else "ai_optout"


ALLOW_ALL = Robots("", "-")
DISALLOW_ALL = Robots("User-agent: *\nDisallow: /", "-")
