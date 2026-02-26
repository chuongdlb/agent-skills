# Environment Header Template

Complete template for `<env_name>.h`:

```cpp
#include <rl_tools/rl/environments/environments.h>

// 1. Parameters — physical constants, limits, timestep
template <typename T>
struct MyEnvParameters {
    // Physics
    constexpr static T DT = 0.05;       // Timestep
    constexpr static T G = 9.81;        // Gravity (if applicable)

    // Limits
    constexpr static T MAX_FORCE = 10;
    constexpr static T MAX_SPEED = 5;

    // Initial state bounds
    constexpr static T INITIAL_STATE_MIN_X = -1;
    constexpr static T INITIAL_STATE_MAX_X = 1;
    constexpr static T INITIAL_STATE_MIN_V = -0.5;
    constexpr static T INITIAL_STATE_MAX_V = 0.5;
};

// 2. Specification — bundles T, TI, PARAMETERS
template <typename T_T, typename T_TI, typename T_PARAMETERS = MyEnvParameters<T_T>>
struct MyEnvSpecification {
    using T = T_T;
    using TI = T_TI;
    using PARAMETERS = T_PARAMETERS;
};

// 3. State — must have static constexpr TI DIM
template <typename T, typename TI>
struct MyEnvState {
    static constexpr TI DIM = 2;
    T x;        // Position
    T x_dot;    // Velocity
};

// 4. Observation — must have static constexpr TI DIM
//    Can differ from state (e.g., Fourier features, partial observability)
template <typename TI>
struct MyEnvObservation {
    static constexpr TI DIM = 3;  // e.g., cos(x), sin(x), x_dot
};

// 5. Environment struct — inherits Environment<T, TI>
template <typename T_SPEC>
struct MyEnv : rl_tools::rl::environments::Environment<typename T_SPEC::T, typename T_SPEC::TI> {
    using SPEC = T_SPEC;
    using T = typename SPEC::T;
    using TI = typename SPEC::TI;

    // Required type aliases
    using Parameters = typename SPEC::PARAMETERS;
    using State = MyEnvState<T, TI>;
    using Observation = MyEnvObservation<TI>;
    using ObservationPrivileged = Observation;  // Same unless asymmetric critic

    // Required constants
    static constexpr TI OBSERVATION_DIM = Observation::DIM;
    static constexpr TI ACTION_DIM = 1;            // Number of control inputs
    static constexpr TI EPISODE_STEP_LIMIT = 200;  // Max steps per episode
};
```

## Notes

- `Parameters` is templated on `T` so constants have correct floating-point type
- `Specification` bundles types to keep the template parameter list manageable
- `State::DIM` is used internally by some rl-tools components
- `Observation::DIM` must match what `observe()` writes (OBSERVATION_DIM = Observation::DIM)
- `ObservationPrivileged` can be a different type for asymmetric actor-critic setups
- `EPISODE_STEP_LIMIT` triggers truncation (not termination) when reached
- `ACTION_DIM` should match the number of columns `step()` expects in the action matrix
