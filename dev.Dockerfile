ARG PLATFORM=linux/amd64

FROM --platform=${PLATFORM} mcr.microsoft.com/devcontainers/python:3.10

COPY . .

RUN pip install -r requirements/requirements.txt \
    -r requirements/dev.txt -r requirements/docs.txt \
    -r requirements/pandas.txt -r requirements/polars.txt \
    -r requirements/transformers.txt \
     && pip install ipykernel

CMD ["python"]
