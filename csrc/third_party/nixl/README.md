# NIXL public headers

The public C++ headers of [NIXL](https://github.com/ai-dynamo/nixl) that the
native NIXL connector (`csrc/storage_backends/nixl/`) compiles against: `nixl.h`
and the three headers it includes. They are copied unmodified from tag
`v1.3.1` (commit `b20f4d0a`), `src/api/cpp/`, and are licensed under the Apache
License 2.0 by NVIDIA, as each file's header states.

They are vendored because the `nixl` wheels ship `libnixl` but no headers. The
connector is compiled against the oldest supported NIXL, so it runs with any
`nixl>=1.3.0,<2`; later NIXL releases have only added to these headers.

To raise the minimum NIXL version, copy the same four files from the new tag and
update the tag above, `NIXL_BUILD_VERSION` in the root `pyproject.toml` (the
`libnixl` the CUDA wheels link against), and the floor in
`requirements/nixl.txt`.
