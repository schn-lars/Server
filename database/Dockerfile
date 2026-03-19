FROM postgres:16

COPY db.dump /docker-entrypoint-initdb.d/
COPY restore.sh /docker-entrypoint-initdb.d/

EXPOSE 5432