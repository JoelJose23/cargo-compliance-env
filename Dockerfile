#1.Selecting a slim version of Python for less space
FROM python:3.10-slim
WORKDIR /app

#2.Copying the dependencies the first thing to increase speed
COPY requirements.txt .

#3.Running to install the python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#4.Copy all the custom modules
COPY . . 

#5.Exposing hugginfaces favourite port
EXPOSE 7860

#6.The command to kick off the FastAPI server
CMD ["uvicorn","server.main:app","--host","0.0.0.0","--port","7860"]