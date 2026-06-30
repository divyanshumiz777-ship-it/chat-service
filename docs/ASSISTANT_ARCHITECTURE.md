# TheSocial Copilot — Assistant Architecture

How the assistant resolves a message into an action, and **how to add a new
command** with a single object.

> TL;DR — to add a command, append **one `Capability`** to
> [`app/llm/capabilities.py`](../app/llm/capabilities.py) and (if it needs a new
> data source) wire its `Intent` to a handler in
> [`app/api/internal.py`](../app/api/internal.py). Nothing else changes.

---

## Request flow

```
POST /internal/v1/chat  { message }      (gateway-only: X-Internal-Token + X-User-Id)
        │
        ▼
  parse_command(message)            ── slash command?  "/friends", "/search users x"
        │  hit → ParsedCommand(intent, term?/query?/types?)
        │  miss
        ▼
  route_intent(message)             ── 1. Capability Registry exact-alias match
        │                              2. regex fallback (legacy long-tail)
        ▼
  dispatch by Intent (api/internal.py)
        ├── MongoDB tools      → deterministic formatter (icons/cards), no LLM
        ├── USER_SEARCH        → ACL-safe people search
        ├── HELP               → render_help_menu()  (from the registry)
        ├── CHAT               → domain-locked LLM (refuses world knowledge)
        └── PLATFORM_SEARCH/QA → RAG (search_all → generate_answer, ACL-filtered)
        │
        ▼
  _attach_suggestions(response, intent)   ── adds related-command hints (Part 5)
```

### Three layers, three jobs

| Layer | File | Responsibility |
|---|---|---|
| **Capability Registry** | `app/llm/capabilities.py` | Single source of truth: every command's aliases, slash forms, icon, category, description, examples, suggestions. Parses slash commands; renders the help menu; serves the palette payload. |
| **Intent Router** | `app/llm/intent_router.py` | Maps a message → `Intent`. Consults the registry first (exact alias), then falls back to regexes for phrasings the registry doesn't enumerate. |
| **Dispatch + tools** | `app/api/internal.py` | Maps `Intent` → handler → MongoDB tool / RAG / LLM, formats the reply deterministically, attaches suggestions. |

The registry is **primary**; the regexes are a **fallback** we keep so no existing
phrasing breaks. Over time, phrasings can migrate from regex to registry aliases.

---

## Why a registry (and not more regexes)

The router used to grow a new regex (and a new `if` in the dispatch, and a new
help string, and a new palette entry) for every command — four edits, four places
to drift. The registry collapses that to **one declarative object** consumed by
routing, slash parsing, help, suggestions, and the frontend palette alike.

---

## How to add a new command

Example: a "my notifications" command backed by a new `get_notifications` tool.

**1. Add the tool** (`app/tools/notifications.py`) — MongoDB only, never raises,
returns `[]` on error. Export it from `app/tools/__init__.py`. *(Skip if reusing
an existing tool.)*

**2. Add an `Intent`** (`app/llm/intent_router.py`) — only if no existing intent
fits:

```python
class Intent(str, Enum):
    ...
    NOTIFICATIONS = "notifications"
```

**3. Register the capability** (`app/llm/capabilities.py`) — the one object:

```python
Capability(
    name="notifications", intent=Intent.NOTIFICATIONS, category="Insights",
    icon="🔔", label="Notifications", command="my notifications",
    description="Your recent notifications.",
    tool="get_notifications",
    aliases=("notifications", "my notifications", "recent notifications", "alerts"),
    slash=("notifications",),
    examples=("my notifications",),
    suggestions=("dashboard", "my friends"),
),
```

**4. Add a handler + dispatch** (`app/api/internal.py`):

```python
async def _notifications(user_id: str, intent: Intent) -> dict:
    items = await get_notifications(user_id)
    if not items:
        return {"answer": _empty_state("No notifications.", ["dashboard", "my friends"]),
                "citations": [], "intent": intent, "notifications": []}
    return {"answer": _format_notifications(items), "citations": [],
            "intent": intent, "notifications": items}

# register in the dispatch table:
_SIMPLE_TOOL_HANDLERS = {
    ...
    Intent.NOTIFICATIONS: _notifications,
}
```

That's it. Aliases route automatically, `/notifications` works, the help menu and
frontend palette list it (it's in `capabilities_payload()`), and a smart-suggestion
footer is appended automatically.

### Capability fields

| Field | Purpose |
|---|---|
| `name` | canonical id (unique) |
| `intent` | the `Intent` it resolves to (→ the dispatch handler / tool) |
| `category` | menu grouping (`CATEGORY_ORDER` controls display order) |
| `icon`, `label`, `description` | menu + palette display |
| `command` | canonical text the palette sends when selected |
| `tool` | tool function name — documentation only |
| `aliases` | **exact, whole-message** NL forms that route here |
| `slash` | slash forms (without the leading `/`) |
| `examples` | shown in the palette / help |
| `suggestions` | related commands appended to responses (Part 5) |
| `hidden` | exclude from menus/palette (still routable by alias) |

---

## Routing guarantees (do-not-break invariants)

- **Exact, whole-message** alias matching only — `"suggest friends"` resolves to
  its own capability and is never swallowed by `"friends"`.
- Anything not an exact alias falls through to the **existing regexes**, so every
  pre-registry phrasing routes exactly as before (see
  `tests/test_capabilities.py::test_existing_routes_preserved`).
- **Domain lock is preserved**: off-topic messages route to `CHAT`, whose
  system prompt refuses world knowledge and redirects to TheSocial. MongoDB tools
  always take priority over RAG.
- **ACL is never bypassed**: `search_users` and `search_all` filter by
  blocked/visibility/participants before anything is formatted.

---

## Frontend integration (palette)

`GET /internal/v1/capabilities` → proxied by Hono as
`GET /api/v1/assistant/capabilities` (`handleGetCapabilities` →
`forwardGetCapabilities`) → returns `capabilities_payload()`.

The palette (`frontend/`) **fetches this once and caches it** (module-level cache
in `hooks/useCommandPalette.ts`), then filters locally on every keystroke
(`lib/commandFilter.ts` — Part 10: no backend call per keystroke). Files:

- `lib/commandFilter.ts` — pure fuzzy match + rank + category grouping (unit-tested
  in `lib/commandFilter.test.ts` via `node --test --experimental-strip-types`).
- `hooks/useCommandPalette.ts` — caps fetch/cache + recent & pinned (localStorage).
- `components/assistant/CommandPalette.tsx` — the floating menu (ARIA listbox,
  keyboard nav, pin toggle, empty state, animations).
- `components/assistant/AssistantInput.tsx` — opens the palette when the input
  starts with `@`; selecting a command **emits its slash form** (`/friends`) to the
  normal `/chat` pipeline. Parametric commands (whose `command` ends in a space,
  e.g. `search users `) are inserted as `/search users ` and wait for the argument
  instead of submitting. There is no second execution path — palette selections and
  typed natural language both POST to `/chat`.

Because the palette is generated from the same registry that powers routing, a new
`Capability` appears in the palette automatically — no frontend change required.
