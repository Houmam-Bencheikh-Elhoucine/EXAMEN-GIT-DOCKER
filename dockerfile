FROM python:3.10-alpine

COPY . /examapp

WORKDIR /examapp

# Install system dependencies required for building Python packages
RUN apk add --no-cache \
    build-base \
    libstdc++ \
    openblas-dev \
    lapack-dev \
    gfortran \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h

# intall python packages
RUN pip3 install -r requirements.txt

CMD python3 train_classifier.py && python3 predict_classifier.py