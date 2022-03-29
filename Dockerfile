FROM python:3.7
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install torch torchvision torchaudio
RUN git clone https://github.com/SKA-INAF/Tiramisu.git /Tiramisu
WORKDIR /Tiramisu
RUN pip install -r requirements.txt
RUN unzip archives/test.zip -d .
ENTRYPOINT [ "python", "inference.py" ]
CMD [ "-i", "test/sample1_galaxy0001.png" ]
