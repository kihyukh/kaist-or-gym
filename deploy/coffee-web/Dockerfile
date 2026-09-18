FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 PORT=8000 COFFEE_DATA_DIR=/data
WORKDIR /app
COPY pyproject.toml README.md ./
COPY kaist_rl_lab ./kaist_rl_lab
COPY deploy/coffee-web/entrypoint.py /usr/local/bin/coffee-web-entrypoint.py
RUN pip install --no-cache-dir '.[web]' \
    && useradd --uid 10001 --create-home coffee \
    && mkdir /data && chown coffee:coffee /data
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=3)" || exit 1
# Initialize the mounted directory, then run the server as the unprivileged user.
ENTRYPOINT ["python", "/usr/local/bin/coffee-web-entrypoint.py"]
CMD ["coffee-pouring-web"]
