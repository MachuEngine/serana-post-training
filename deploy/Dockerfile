# vLLM's official OpenAI-compatible server image, pinned.
#
# Pinned deliberately: P5's measurements (KV-cache blocks, the
# throughput/concurrency knee, AWQ vs bf16) were taken against an
# *unpinned* `pip install vllm` on the VM, so the serving engine version
# was never recorded alongside the numbers. This image is the first time
# it is. Changing this tag invalidates comparability with those tables --
# treat it the way DESIGN.md §9.4 treats the judge model.
FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

# Only what the server needs: the config tree (base.yaml's serving knobs,
# and the experiment files serve_up.py enumerates to find adapter ids),
# the config loader, and the launcher. No training code, no data, no
# adapters, no secrets -- see .dockerignore, which keeps them out of the
# build context entirely rather than relying on COPY discipline.
COPY config/ /app/config/
COPY src/__init__.py src/config.py /app/src/
COPY scripts/serve_up.py /app/scripts/serve_up.py

# Adapters are NOT baked in. An initContainer pulls them from GCS into
# this path at pod start (deploy/k8s/deployment.yaml). Two reasons:
# a new adapter must not require a new image, and the adapters are
# derived from Bethesda-owned dialogue -- keeping them out of the image
# keeps them off the registry.
ENV ADAPTER_ROOT=/adapters
# Model weights download here. Backed by an emptyDir, so a pod restart
# re-downloads the 16 GB base model -- see the startup probe.
ENV HF_HOME=/hf-cache

# The base image's entrypoint is the API server itself, which would take
# its serving knobs from the k8s manifest and duplicate base.yaml. Replace
# it with the config-driven launcher so the image has no serving knobs of
# its own (CLAUDE.md: no hardcoded hyperparameters outside config/).
ENTRYPOINT ["python3", "/app/scripts/serve_up.py"]
