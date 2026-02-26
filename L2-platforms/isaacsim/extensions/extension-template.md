# Extension Templates

Copy-paste scaffolds for common extension types in Isaac Sim.

## Minimal Python-Only Extension

### config/extension.toml
```toml
[package]
version = "0.1.0"
category = "Simulation"
title = "My Extension"
description = "Description of the extension."
keywords = ["isaac"]
changelog = "docs/CHANGELOG.md"
readme = "docs/README.md"
preview_image = "data/preview.png"
icon = "data/icon.png"
writeTarget.kit = true

[dependencies]
"isaacsim.core.api" = {}

[[python.module]]
name = "isaacsim.my_domain.my_feature"

[[test]]
timeout = 600
args = [
    "--enable", "omni.kit.loop-isaac",
    "--reset-user",
    "--vulkan",
    "--/app/asyncRendering=false",
    "--no-window",
]
```

### python/impl/__init__.py
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from .extension import Extension
```

### python/impl/extension.py
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import omni.ext


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        pass

    def on_shutdown(self):
        pass
```

## Extension with OmniGraph Nodes

### config/extension.toml (additions)
```toml
[dependencies]
"isaacsim.core.api" = {}
"omni.graph" = {}

[fswatcher.patterns]
include = ["*.ogn", "*.py", "*.toml"]
exclude = ["Ogn*Database.py"]
```

### python/nodes/OgnMyNode.ogn
```json
{
    "MyNodeName": {
        "version": 1,
        "description": "Description of what this node computes",
        "language": "Python",
        "categories": {
            "isaacMyDomain": "Brief category description"
        },
        "metadata": {
            "uiName": "My Node Display Name"
        },
        "inputs": {
            "execIn": {
                "type": "execution",
                "description": "The input execution trigger"
            },
            "prim": {
                "type": "target",
                "description": "Target prim path",
                "optional": true
            },
            "value": {
                "type": "double",
                "description": "Input value",
                "default": 0.0
            }
        },
        "outputs": {
            "execOut": {
                "type": "execution",
                "description": "Output execution trigger"
            },
            "result": {
                "type": "double",
                "description": "Computed result",
                "default": 0.0
            }
        }
    }
}
```

### python/nodes/OgnMyNode.py
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from omni.graph.core import Database


class OgnMyNode:
    @staticmethod
    def compute(db: Database) -> bool:
        try:
            value = db.inputs.value
            db.outputs.result = value * 2.0
        except Exception as e:
            db.log_error(f"Error in compute: {e}")
            return False
        return True
```

## Extension with C++ Plugin

### config/extension.toml (additions)
```toml
[[native.plugin]]
path = "bin/*.plugin"
recursive = false
```

### python/impl/__init__.py
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from .extension import Extension
```

### python/impl/extension.py
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import carb
import omni.ext

from ..bindings._isaacsim_my_ext import acquire_interface, release_interface


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.__interface = acquire_interface()

    def on_shutdown(self):
        release_interface(self.__interface)
        self.__interface = None
```

### plugins/ directory

C++ plugin source files go here. The build system (premake5) compiles them into `bin/*.plugin`.

## Extension with UI Components

### config/extension.toml (additions)
```toml
[dependencies]
"isaacsim.core.api" = {}
"omni.kit.uiapp" = {}
"omni.ui" = {}
"omni.kit.menu.utils" = {}
```

### python/impl/extension.py
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import gc

import omni.ext
import omni.ui as ui
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._menu_items = [
            MenuItemDescription(
                name="Tools/My Tool",
                onclick_fn=lambda *_: self._show_window(),
            )
        ]
        add_menu_items(self._menu_items, "Tools")
        self._window = None

    def on_shutdown(self):
        remove_menu_items(self._menu_items, "Tools")
        if self._window:
            self._window.destroy()
            self._window = None
        gc.collect()

    def _show_window(self):
        if self._window is None:
            self._window = ui.Window("My Tool", width=400, height=300)
            with self._window.frame:
                with ui.VStack():
                    ui.Label("My Tool UI")
                    ui.Button("Do Something", clicked_fn=self._on_click)

    def _on_click(self):
        pass
```
