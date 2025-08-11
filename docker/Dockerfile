FROM openfoam/openfoam11-paraview:latest

# Install Python and development tools
RUN apt-get update && \
    apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install uv for fast package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install project dependencies
RUN uv pip install --system -e .

# Set OpenFOAM environment
ENV PATH="/opt/openfoam11/platforms/linux64GccDPInt32Opt/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/openfoam11/platforms/linux64GccDPInt32Opt/lib:${LD_LIBRARY_PATH}"
ENV WM_PROJECT_DIR="/opt/openfoam11"

# Source OpenFOAM bashrc on container start
RUN echo "source /opt/openfoam11/etc/bashrc" >> /root/.bashrc

# Set entrypoint
ENTRYPOINT ["/bin/bash", "-c", "source /opt/openfoam11/etc/bashrc && exec \"$@\"", "--"]
CMD ["/bin/bash"]