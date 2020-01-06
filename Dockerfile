FROM tensorflow/tensorflow:latest-py3-jupyter

MAINTAINER lolik samuel

WORKDIR /app

#RUN apt-get update  && apt-get install -y build-essential mpich libpq-dev

#RUN conda install pip

# RUN pip install --upgrade pip \
#  && pip install -r requirements.txt

# Update aptitude with new repo
RUN apt-get update

# Install software
RUN apt-get install -y \
    git

#      && anaconda-navigator=1.9.7 \
#      && anaconda-project=0.8.3 \
#      && conda=4.8.0
#         conda-build==3.18.8
#         conda-package-handling==1.6.0
#         conda-verify==3.4.2
#         Bottleneck==1.3.1
#         mkl-random==1.0.1.1
#         mkl-service==2.3.0
#         navigator-updater==0.2.1
#         pycurl==7.43.0.3
#         xlwings==0.16.3

# Clone the conf files into the docker container
#RUN git clone https://github.com/loliksamuel/py-ML-rl-trade.git
# WORKDIR py-ML-rl-trade
COPY *.py /app/
COPY files /app/files
COPY requirements.txt /app/



RUN pip install -r requirements.txt

# CMD ["python","rl_dqn.py"]
ENTRYPOINT ["python","./rl_dqn.py" ]



# python               rl_dqn.py   -na 'test_sinus' -ne 20 -nf 20 -nn 64 -nb 20
# docker run --rm <yourImageName>  -na 'test_sinus' -ne 20 -nf 20 -nn 64 -nb 20





