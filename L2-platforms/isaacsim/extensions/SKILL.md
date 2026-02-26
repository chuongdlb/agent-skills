---
name: isaacsim-extensions
description: >
  Creates and modifies Omniverse Kit extensions for Isaac Sim — extension.toml manifests, Python module scaffolding, OmniGraph nodes, C++ plugins.
layer: L2
domain: [robotics, simulation]
source-project: IsaacSim
depends-on: [isaacsim-simulation-core, isaacsim-build-and-test]
tags: [extensions, omnigraph, kit, plugins]
---

# Developing Isaac Sim Extensions

Isaac Sim extensions live under `source/extensions/` and follow Omniverse Kit SDK conventions. Each extension is a self-contained unit with configuration, Python modules, optional C++ plugins, and test definitions.

## Quick Start: Minimal Extension

```
source/extensions/isaacsim.my_domain.my_feature/
  config/
    extension.toml          # Extension manifest (required)
  python/
    impl/
      __init__.py           # Exports Extension class
      extension.py          # on_startup / on_shutdown hooks
  data/
    icon.png
    preview.png
  docs/
    CHANGELOG.md
    README.md
```

## Naming Convention

Extensions use `isaacsim.<domain>.<feature>` dot-separated naming:

| Domain | Examples |
|--------|----------|
| core | isaacsim.core.api, isaacsim.core.prims, isaacsim.core.utils |
| sensors | isaacsim.sensors.camera, isaacsim.sensors.physics, isaacsim.sensors.rtx |
| robot | isaacsim.robot.manipulators, isaacsim.robot.wheeled_robots |
| robot_motion | isaacsim.robot_motion.motion_generation, isaacsim.robot_motion.lula |
| robot_setup | isaacsim.robot_setup.wizard, isaacsim.robot_setup.gain_tuner |
| ros2 | isaacsim.ros2.bridge, isaacsim.ros2.sim_control |
| asset | isaacsim.asset.importer.urdf, isaacsim.asset.exporter.urdf |
| replicator | isaacsim.replicator.examples, isaacsim.replicator.writers |
| gui | isaacsim.gui.menu, isaacsim.gui.components |
| util | isaacsim.util.debug_draw, isaacsim.util.physics |

UI-specific extensions append `.ui` (e.g., `isaacsim.robot.manipulators.ui`). Example/demo extensions append `.examples`.

## Extension Directory Layout

```
isaacsim.<domain>.<feature>/
  config/
    extension.toml              # Manifest: metadata, dependencies, modules, tests
    CategoryDefinition.json     # OmniGraph category definitions (if using OGN)
  python/
    impl/                       # Private implementation
      __init__.py               # Exports Extension class
      extension.py              # IExt lifecycle hooks
      <feature_modules>.py
    nodes/                      # OmniGraph node implementations
      OgnMyNode.ogn             # Node definition (JSON)
      OgnMyNode.py              # Node compute logic
    tests/                      # Unit tests
      __init__.py
      test_*.py
  plugins/                      # C++ plugins (optional)
    bindings/
      _isaacsim_my_ext.pyx
  data/                         # Assets, icons
    icon.png
    preview.png
  docs/
    CHANGELOG.md
    README.md
```

## Python Extension Lifecycle

All extensions implement `omni.ext.IExt`:

```python
import omni.ext

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Called when the extension is loaded. ext_id includes version."""
        pass

    def on_shutdown(self):
        """Called when the extension is unloaded. Release all resources."""
        pass
```

### Common Patterns

**Minimal (no-op startup):**
```python
class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        pass
    def on_shutdown(self):
        pass
```

**Resource acquisition (C++ interface):**
```python
class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.__interface = acquire_interface()
        self.registered_items = []
        try:
            self.register_items()
        except Exception as e:
            carb.log_error(f"Could not register items: {e}")

    def on_shutdown(self):
        release_interface(self.__interface)
        self.__interface = None
```

**Async initialization:**
```python
class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._settings = carb.settings.get_settings()
        self.__task = asyncio.ensure_future(self.__async_setup())

    async def __async_setup(self):
        await omni.kit.app.get_app().next_update_async()
        # Async init logic here
```

**Sub-component pattern:**
```python
class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.__component = MyComponent(ext_id)

    def on_shutdown(self):
        self.__component.shutdown()
        del self.__component
        gc.collect()
```

## OmniGraph Node Pattern (.ogn)

OGN files are JSON definitions for OmniGraph computation nodes:

```json
{
    "MyNodeName": {
        "version": 1,
        "description": "Description of the node",
        "language": "Python",
        "categories": {
            "isaacCategory": "Brief category description"
        },
        "metadata": {
            "uiName": "Human-Readable Node Name"
        },
        "inputs": {
            "execIn": {
                "type": "execution",
                "description": "Execution trigger"
            },
            "myInput": {
                "type": "double",
                "description": "A numeric input",
                "default": 0.0
            },
            "prim": {
                "type": "target",
                "description": "Target prim path",
                "optional": true
            }
        },
        "outputs": {
            "execOut": {
                "type": "execution",
                "description": "Output execution trigger"
            },
            "result": {
                "type": "float[]",
                "description": "Computed result values"
            }
        }
    }
}
```

**Common OGN types:** `execution`, `target`, `float`, `double`, `bool`, `int`, `token`, `float[]`, `double[]`, `token[]`, `int[]`.

The Python compute implementation lives alongside in `OgnMyNode.py`.

## Dependency Declaration

In `extension.toml`, the `[dependencies]` section declares required and optional extensions:

```toml
[dependencies]
"isaacsim.core.api" = {}                    # Required dependency
"omni.kit.material.library" = {optional = true}  # Optional dependency
"omni.physx.tensors" = {}
```

All Isaac Sim extensions are discovered by Kit SDK via the extension search paths configured in the app `.kit` files and `repo.toml`.

## Extension Registration

Extensions are discovered automatically by Kit SDK when their parent directory is included in the extension search paths. No manual registration step is needed beyond placing the extension under `source/extensions/`.

## Reference Files

- [extension-toml-reference.md](extension-toml-reference.md) - Complete annotated extension.toml schema
- [extension-template.md](extension-template.md) - Copy-paste extension scaffolds

## Key Repo Examples

| Extension | Pattern | Path |
|-----------|---------|------|
| Core API (large, Python+C++) | `source/extensions/isaacsim.core.api/` |
| Camera Sensor | `source/extensions/isaacsim.sensors.camera/` |
| ROS2 Bridge (complex, multi-module) | `source/extensions/isaacsim.ros2.bridge/` |
| Gripper OGN Node | `source/extensions/isaacsim.robot.manipulators/python/nodes/OgnIsaacGripperController.ogn` |
| Deprecation Manager | `source/extensions/isaacsim.core.deprecation_manager/` |
