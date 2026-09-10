#!/usr/bin/env bash
set -euo pipefail

command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash -s stable

mkdir -p ~/.claude
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path.home() / ".claude" / "settings.json"
s = json.loads(p.read_text()) if p.exists() else {}
s["theme"] = "auto"
p.write_text(json.dumps(s, indent=2) + "\n")
EOF

bash scripts/common/install_hooks.sh

bash scripts/devcontainer/install_wasm_bindgen.sh
