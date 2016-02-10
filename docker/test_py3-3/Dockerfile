FROM python:3.3
MAINTAINER steven.e.cutting@linux.com

RUN pip install --upgrade pip
RUN pip install pytest toolz pytest-runner

VOLUME /app

WORKDIR /app

CMD ["/app/docker/test_py2/docker_pytest.sh"]
