
# docker build -t devcontainer:latest -f .\.devcontainer\Dockerfile .
FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive

# Latex (from https://github.com/blang/latex-docker/blob/master/Dockerfile.ubuntu)
RUN apt-get update && apt-get install -qy build-essential wget libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -q && apt-get install -qy \
    texlive-full \
    python-pygments gnuplot \
    make git \
    && rm -rf /var/lib/apt/lists/*

# install basic stuff
RUN apt-get update && apt-get -qy upgrade \
    &&  apt-get install -qy \
    apt-utils git curl git cmake sl sudo net-tools nmap file \
    iputils-ping

# install python stuff
RUN apt-get install -y python3.8 

# set py 3.8 to default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --config python3
# install pip
RUN apt-get install -y python3-pip 



# to fix annoying pip xserver bug (https://github.com/pypa/pip/issues/8485)
RUN printf "%s\n" "alias pip3='DISPLAY= pip3'" "alias python=python3" > ~/.bash_aliases

# install packages
RUN pip3 install --upgrade pip 
RUN pip3 install \
    numpy \
    matplotlib \
    pylint \
    autopep8 \
    sympy \
    jupyter \
    pandas
RUN pip3 install \
    numba \
    torch \
    opencv-contrib-python \
    networkx \
    plotly \
    dash \
    dash_bootstrap_components 


# Set some stuff
# RUN git config --global user.email "emil.martens@gmail.com" && git config --global user.name "Emil Martens"

COPY .gitconfig ~
