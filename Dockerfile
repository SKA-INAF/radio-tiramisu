FROM ubuntu:16.04


######################################
##   DEFINE CUSTOMIZABLE ARGS/ENVS
######################################
ARG USER_ARG="caesar"
ENV USER $USER_ARG

ENV PYTHONPATH_BASE ${PYTHONPATH}

##########################################################
##     INSTALL SYS LIBS (IF NOT PRESENT IN BASE IMAGE
##########################################################

# - Install OS packages
RUN apt-get update && apt-get install -y software-properties-common apt-utils curl binutils libtool pkg-config build-essential autoconf automake debconf-utils software-properties-common dpkg-dev git cmake wget bzip2 nano unzip locate less ca-certificates iputils-ping nmap dnsutils

# - Reinstall
RUN apt-get install --reinstall python3-pkg-resources

# - Install python3.6
RUN unset PYTHONPATH && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev 

# - Install pip3.6
RUN unset PYTHONPATH && curl https://bootstrap.pypa.io/get-pip.py | python3.6

# - Make python3.6 as the default
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

# - Install packages
RUN apt-get update && apt-get --no-install-recommends install -y libcurl3 openssl libssl-dev uuid-dev libcap-dev libpcre3-dev util-linux openssh-client openssh-server libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

# - Install git-lsf (needed otherwise large hdf5 data in repo are given a wrong signature and failed to be read)
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get update && apt-get install git-lfs

##########################################################
##     CREATE USER
##########################################################
# - Create user & set permissions
RUN adduser --disabled-password --gecos "" $USER && \
    mkdir -p /home/$USER && \
    chown -R $USER:$USER /home/$USER


######################################
##     INSTALL TIRAMISU
######################################

ENV TIRAMISU_TOP_DIR /opt/Software/Tiramisu
ENV TIRAMISU_DIR $TIRAMISU_TOP_DIR/install


FROM continuumio/miniconda3

WORKDIR $TIRAMISU_TOP_DIR

# Create the environment:
COPY tiramisu_environment.yml .
RUN conda env create -f tiramisu_environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "tiramisu", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
COPY ./test/test.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tiramisu", "python", "./test/test.py"]


# - Clone tiramisu
RUN mkdir -p $TIRAMISU_TOP_DIR $TIRAMISU_DIR $TIRAMISU_DIR/share $TIRAMISU_DIR/lib/python3.6/site-packages
RUN cd $TIRAMISU_TOP_DIR && git clone https://github.com/SKA-INAF/Tiramisu.git