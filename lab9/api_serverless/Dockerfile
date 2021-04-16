# Starting from an official AWS image
FROM amazon/aws-lambda-python:3.6

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN sed -i 's/cu101/cpu/' requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

# Copy only the relevant directories and files
COPY text_recognizer/ ./text_recognizer
COPY api_serverless/app.py ./app.py

CMD ["app.handler"]
