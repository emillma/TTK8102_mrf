
# docker build -t devcontainer:latest -f .\.devcontainer\Dockerfile .
FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates
# RUN update-ca-certificates
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

COPY .gitconfig ~
