FROM python:3.10-slim

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libgomp1 libquadmath0
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/ml/processing

ENTRYPOINT ["python"]