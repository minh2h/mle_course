FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
VOLUME /app
ENV JUPYTER_ENABLE_LAB=yes
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8000", "--no-browser", "--allow-root", "--notebook-dir"]