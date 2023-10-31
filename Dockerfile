FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8.10
ENV TZ=Asia/Tokyo

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # for usability
        tzdata default-jre \
        # developing tools
        vim wget curl git cmake bash-completion \
        # for Python and pip(from https://devguide.python.org/setup/#linux)
        build-essential gdb lcov pkg-config \
        libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
        libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
        lzma lzma-dev tk-dev uuid-dev zlib1g-dev \
        # for poetry
        python3.8 libreadline-dev llvm libncursesw5-dev xz-utils python-openssl python3-distutils \
    # clean apt cache
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo 'source /etc/bash_completion\n' >> /root/.bashrc

# poetry install (use 3.8 to install latest poetry)
RUN curl -sSL https://install.python-poetry.org | python3.8 -
ENV PATH /root/.local/bin:$PATH

# pyenv install
RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv \
    && echo 'PYENV_ROOT=$HOME/.pyenv\nPATH=$PYENV_ROOT/bin:$PATH\neval "$(pyenv init -)"\neval "$(pyenv init --path)"\n' >> /root/.activate_pyenv \
    && echo 'source /root/.activate_pyenv\n' >> /root/.bashrc

WORKDIR /src
COPY pyproject.toml poetry.lock ./

# poetry modules install
RUN . /root/.activate_pyenv \
    && CONFIGURE_OPTS=--enable-shared pyenv install ${PYTHON_VERSION} \
    && pyenv shell ${PYTHON_VERSION} \
    && poetry env use ${PYTHON_VERSION} \
    && poetry install
