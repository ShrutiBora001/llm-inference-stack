# Built on NGC PyTorch: it ships a CUDA-matched torch, NCCL, and Nsight Systems.
# Never install plain `torch` on a GPU image -- the wheel must match the driver.
ARG NGC_TAG=25.01-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

# NGC images mark the system interpreter externally managed (PEP 668), so we
# create a venv that INHERITS the container's tuned torch rather than pulling a
# fresh 2 GB copy. Learned the hard way on Vast #43149908.
ENV VIRTUAL_ENV=/opt/lis-venv
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace/lis
COPY pyproject.toml README.md ./
COPY src/ ./src/
# dattn is the frozen upstream finding; vendored so the image is self-contained
COPY vendor/dattn/ ./vendor/dattn/
RUN pip install --no-cache-dir -e ./vendor/dattn -e ".[dev]"

COPY tests/ ./tests/
COPY bench/ ./bench/
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Fail fast at build time if the package cannot even import.
RUN python -c "import lis, dattn; print('lis + dattn import OK')"

ENV PYTHONUNBUFFERED=1 \
    LIS_WORLD_SIZE=4 \
    LIS_SEQ=32768 \
    LIS_LAYOUT=striped \
    LIS_BLOCK=512

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["validate"]
