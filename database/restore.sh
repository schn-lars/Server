#!/bin/bash
set -e

echo "Creating birdie database (if not exists)"

psql -U thesis -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'birdie'" | grep -q 1 || \
psql -U thesis -d postgres -c "CREATE DATABASE birdie;"

echo "Restoring database from dump..."

pg_restore -U "$POSTGRES_USER" \
           -d birdie \
           --no-owner \
           /docker-entrypoint-initdb.d/db.dump

echo "Done."