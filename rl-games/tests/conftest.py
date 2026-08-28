"""pytest 启动前的公共配置。

作用:让 `import shared`、`import snake` 在任意目录运行 pytest 时都能生效。
原理:把项目根目录(rl-games)插到 sys.path 最前面。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
