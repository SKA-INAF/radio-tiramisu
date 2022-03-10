FROM python:3.7
WORKDIR /app
RUN pip3 install torch torchvision torchaudio
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . /app/
RUN unzip /app/archives/test.zip
ENTRYPOINT [ "python", "inference.py" ]
CMD [ "-i", "test/sample1_galaxy0001.png" ]