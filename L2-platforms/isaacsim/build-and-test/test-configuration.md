# Test Configuration Reference

Complete test setup reference for Isaac Sim extensions.

## Full Annotated [[test]] Section

```toml
# Default test suite (unnamed)
[[test]]
timeout = 900                    # Maximum test duration in seconds

# Extensions loaded alongside this extension during testing
dependencies = [
    "isaacsim.test.utils",       # Test utilities
    "omni.kit.renderer.core",   # Renderer (needed for visual tests)
    "isaacsim.core.cloner",     # Environment cloning (if needed)
]

# Include/exclude specific Python tests
pythonTests.include = ["*"]     # Default: run all
pythonTests.exclude = []        # Exclude patterns

# C++ test libraries
cppTests.libraries = []         # e.g., ["bin/${lib_prefix}myext.tests${lib_ext}"]

# Enable/disable this test suite
enabled = true                  # Default: true

# Error pattern exclusions (known non-fatal errors)
stdoutFailPatterns.exclude = [
    "*[Error] [carb] [Plugin: omni.sensors.nv.lidar.ext.plugin]*",
    "*[Error] [rtx.postprocessing.plugin]*",
]

# Runtime arguments
args = [
    # === Required for Isaac Sim tests ===
    "--enable", "omni.kit.loop-isaac",  # Isaac Sim loop
    "--reset-user",                      # Reset user settings
    "--vulkan",                          # Use Vulkan renderer

    # === Rendering ===
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",

    # === Application ===
    "--/app/file/ignoreUnsavedOnExit=true",
    "--/app/settings/persistent=false",

    # === Fabric ===
    "--/app/settings/fabricDefaultStageFrameHistoryCount=3",

    # === Physics ===
    "--/app/settings/persistent/physics/updateToUsd=false",
    "--/app/settings/persistent/physics/useFastCache=false",
    "--/app/settings/persistent/physics/numThreads=1",
    "--/app/settings/persistent/physics/updateTransformationsFromSDK=false",
    "--/app/settings/persistent/physics/updateVelocitiesToUsd=false",
    "--/app/settings/persistent/simulation/minFrameRate=15",

    # === Assets ===
    "--/persistent/isaac/asset_root/default=/isaac-sim-assets/default",
    "--/persistent/isaac/asset_root/nvidia=/isaac-sim-assets/nvidia",

    # === Viewport ===
    "--/app/renderer/resolution/width=64",    # Low res for speed
    "--/app/renderer/resolution/height=64",

    # === Miscellaneous ===
    "--/persistent/app/omniverse/gamepadCameraControl=false",
    "--no-window",                            # Headless mode
]
```

## Standard Test Args Template

Copy this block for new extension tests:

```toml
[[test]]
timeout = 600
dependencies = [
    "isaacsim.test.utils",
]
args = [
    "--enable", "omni.kit.loop-isaac",
    "--reset-user",
    "--vulkan",
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",
    "--/app/file/ignoreUnsavedOnExit=true",
    "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
    "--/app/settings/persistent=false",
    "--/app/settings/persistent/physics/updateToUsd=false",
    "--/app/settings/persistent/physics/useFastCache=false",
    "--/app/settings/persistent/physics/numThreads=1",
    "--/app/settings/persistent/simulation/minFrameRate=15",
    "--/persistent/isaac/asset_root/default=/isaac-sim-assets/default",
    "--/persistent/isaac/asset_root/nvidia=/isaac-sim-assets/nvidia",
    "--/app/renderer/resolution/width=64",
    "--/app/renderer/resolution/height=64",
    "--/persistent/app/omniverse/gamepadCameraControl=false",
    "--no-window",
]
stdoutFailPatterns.exclude = []
```

## Named Test Suites

Define multiple test suites with different configurations:

```toml
# Main test suite (default, no name)
[[test]]
timeout = 900
args = [...]

# Startup smoke test
[[test]]
name = "startup"
timeout = 300
args = [
    "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
]

# C++ doc tests
[[test]]
name = "doctest"
enabled = true
timeout = 900
pythonTests.include = []
pythonTests.exclude = ["*"]
cppTests.libraries = ["bin/${lib_prefix}myext.tests${lib_ext}"]
```

## Error Pattern Exclusion Syntax

```toml
stdoutFailPatterns.exclude = [
    # Wildcard matching
    "*[Error] [carb]*",

    # Specific error messages
    "*Invalid articulation pointer for*",

    # Plugin dependency errors
    "*[Error] [carb] [Plugin: omni.sensors.nv.lidar.ext.plugin] Dependency: [omni::sensors::lidar::IGenericModelOutputIOFactory v0.1] failed*",

    # RTX rendering errors
    "*[Error] [rtx.postprocessing.plugin] DepthSensor: Texture sizes do not match*",
]
```

Use `*` as wildcard. Each pattern is matched against stdout/stderr lines. If a line matches any exclude pattern, it won't trigger a test failure.

## Extension-Specific Test Args

Some extensions need extra settings:

```toml
# ROS2 bridge
args = [
    # ... standard args ...
    '--/exts/isaacsim.ros2.bridge/ros_distro="system_default"',
    "--/exts/isaacsim.ros2.bridge/publish_without_verification=1",
]

# Sensor extensions (may need higher resolution)
args = [
    # ... standard args ...
    "--/app/renderer/resolution/width=256",
    "--/app/renderer/resolution/height=256",
]
```

## Test Dependencies

```toml
dependencies = [
    # Common test utilities
    "isaacsim.test.utils",

    # Renderer (needed for visual/render tests)
    "omni.kit.renderer.core",

    # Environment cloning
    "isaacsim.core.cloner",

    # Debug visualization
    "isaacsim.util.debug_draw",

    # UI testing
    "omni.graph.ui",

    # Wheeled robots (for navigation tests)
    "isaacsim.robot.wheeled_robots",
]
```

## Running Tests Locally

```bash
# Run all tests for a specific extension
./repo.sh test -e isaacsim.my_domain.my_feature

# Run a specific test suite
./repo.sh test --suite pythontests

# Run with name filter
./repo.sh test -f test_my_function

# Run startup tests only
./repo.sh test --suite startuptests

# Run benchmark tests
./repo.sh test --suite benchmarks

# Run with verbose output
./repo.sh test -e isaacsim.my_domain.my_feature -v
```

## Debugging Test Failures

### Common Failure Causes

1. **Timeout:** Increase `timeout` in `[[test]]` section
2. **Missing dependency:** Add to `dependencies` list
3. **Known error in stdout:** Add pattern to `stdoutFailPatterns.exclude`
4. **Rendering issues:** Ensure `--vulkan` and correct resolution args
5. **Physics determinism:** Set `--/app/settings/persistent/physics/numThreads=1`
6. **Asset loading:** Verify asset root paths in args

### Test File Location

Test files go in the extension's test module:
```
source/extensions/isaacsim.my_domain.my_feature/
  python/
    tests/
      __init__.py
      test_basic.py
      test_advanced.py
```

Register the test module in `extension.toml`:
```toml
[[python.module]]
name = "isaacsim.my_domain.my_feature.tests"
public = false
```

### Test Structure

```python
# test_basic.py
import omni.kit.test

class TestMyFeature(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        """Set up test fixtures."""
        await omni.usd.get_context().new_stage_async()

    async def tearDown(self):
        """Clean up."""
        pass

    async def test_basic_function(self):
        """Test basic functionality."""
        # Test implementation
        self.assertTrue(result)
        self.assertEqual(expected, actual)
```

## CI Test Matrix

From `.github/workflows/build-and-test.yml`:

- Platform: Linux x86_64 (self-hosted GPU runners)
- Timeout: 240 minutes total
- Build: `./build.sh`
- Tests: startup, warmup, full test suites
- Cache clearing between runs (Omniverse, Packman)
