# Mini glossary — tools & MCP (Session 03 vocabulary)

- **Tool declaration**: name + natural-language description + JSON Schema for parameters.
- **Request-execute-append cycle**: messages + schemas go to the model; it emits a
  tool_use block; the runtime executes; the result is appended; repeat until plain text.
- **Tool-choice modes**: auto, required/any, specific tool, none.
- **MCP host / client / server**: the LLM app / one stateful session per server /
  the process exposing tools, resources and prompts.
- **Discovery**: `tools/list` returns tool schemas at runtime; `tools/call` invokes.
- **Security boundary**: the model proposes; the runtime disposes.
