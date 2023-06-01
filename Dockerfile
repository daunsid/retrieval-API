FROM python:3.

#RUN useradd -ms /bin/bash

WORKDIR /usr/src/app
EXPOSE 8000



COPY . .
# install pipenv and project dependencies
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile



ENTRYPOINT [ "uvicorn" ]
CMD [ "main:app", "--host", "0.0.0.0" ]