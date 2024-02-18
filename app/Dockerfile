FROM python:3.10.12-alpine

WORKDIR /web

EXPOSE 80

COPY . /web/

RUN pip install --upgrade pip

RUN pip install -r /web/requirements.txt

CMD streamlit run main.py --server.port 80 

