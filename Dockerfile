FROM tensorflow/tensorflow:latest-py3-jupyter

MAINTAINER lolik samuel

#ADD ./requirements.txt /app/

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
RUN git clone https://github.com/loliksamuel/py-ML-rl-trade.git
# COPY *.py /app/
# COPY files /app/files
WORKDIR py-ML-rl-trade

RUN pip install -r requirements.txt

#CMD [ "python", "./rl_dqn.py" ]




#how to run ?
#----------------------
#1. docker build -t app:1.0 .
#2. wait few minutes on 1st run(later it will be faster)
#3. docker run -it <image_id> /bin/bash
#4. python rl_dqn.py   -na 'test_sinus' -ne 2000 -nf 20 -nn 64 -nb 20
#5. python backtest.py -na 'test_sinus' -mn 'model_ep2000' -tf 0.
#6. expect to earn x$
#7. python rl_dqn.py   -na '^GSPC_2011' -ne 20000 -nf 20 -nn 64 -nb 20
#8. python backtest.py -na '^GSPC_2019' -mn 'model_ep20000' -tf 0.
#9. expect to earn x$
#10. docker run app:1.0 .

