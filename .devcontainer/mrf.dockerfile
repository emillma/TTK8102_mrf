
# docker build -t devcontainer:latest -f .\.devcontainer\Dockerfile .
FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y apt-utils git

# Latex (from https://github.com/blang/latex-docker/blob/master/Dockerfile.ubuntu)
RUN apt-get update && apt-get install -qy build-essential wget libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -q && apt-get install -qy \
    texlive-full \
    python-pygments gnuplot \
    make git \
    && rm -rf /var/lib/apt/lists/*

# install python stuff
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.9

# set py 3.8 to default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --config python3
# install pip
RUN apt-get update && apt-get install -y python3-pip 


# install packages
RUN pip3 install \
    numpy scipy matplotlib pyqt5 pandas\
    pylint autopep8 jupyter \
    sympy 
RUN pip3 install \
    plotly dash\
    numba \
    torch torchvision\
    opencv-contrib-python \
    networkx \
    dash_bootstrap_components 

# to fix annoying pip xserver bug (https://github.com/pypa/pip/issues/8485)
RUN printf "%s\n" "alias pip3='DISPLAY= pip3'" "alias python=python3" > ~/.bash_aliases

# RUN git config --global user.email "emil.martens@gmail.com" && git config --global user.name "Emil Martens"

COPY .gitconfig ~
