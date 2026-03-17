FROM postgres:latest

COPY mydump.dump /docker-entrypoint-initdb.d/
COPY restore.sh /docker-entrypoint-initdb.d/

EXPOSE 5432