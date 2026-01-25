FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install dependencies
COPY requirements.txt /job/
RUN pip install --no-cache-dir -r requirements.txt
