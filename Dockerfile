FROM library/ubuntu:16.04
RUN apt-get -y update
RUN apt-get install -y build-essential libbz2-dev libssl-dev libreadline-dev libsqlite3-dev tk-dev
RUN apt-get install -y libpng-dev libfreetype6-dev
RUN apt-get install -y python3 libpython3-dev python3-dev python3-pip
RUN pip3 install numpy scipy pandas
RUN pip3 install spacy
RUN pip3 install nltk
RUN pip3 install tqdm
RUN pip3 install networkx

RUN python3 -m nltk.downloader stopwords
RUN python3 -m spacy.en.download all

COPY ["analyze.py", "analyze.py"]
COPY ["runner.py", "runner.py"]
COPY ["n100 group novice.csv", "raw_samples.csv"]

CMD ["python3", "runner.py"]
