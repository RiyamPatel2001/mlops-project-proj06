---
name: Docker config.yaml is always a runtime mount
description: config.yaml is never baked into Docker images in this project — always mounted at runtime
type: feedback
---

config.yaml is always mounted at runtime via `-v`, never part of the Docker build context.

**Why:** The project intentionally keeps config separate from the image so hyperparameters and URIs can be changed without rebuilding.

**How to apply:** Never describe config.yaml as "baked in" or say a runtime mount "overrides" it. Just show the `-v config.yaml:/app/training/config.yaml` mount as the normal required step.
