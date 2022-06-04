FROM python:3.8
WORKDIR /app
COPY . .
RUN pip freeze
RUN pip3 install numpy
RUN pip install Flask Jinja2
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
