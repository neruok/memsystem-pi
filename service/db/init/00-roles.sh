#!/bin/sh
set -eu

psql --set=ON_ERROR_STOP=1 --set=service_password="$MEMSYSTEM_SERVICE_PASSWORD" \
  --set=database="$POSTGRES_DB" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE ROLE memsystem_owner NOLOGIN;
CREATE ROLE memsystem_service LOGIN PASSWORD :'service_password'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
GRANT CONNECT ON DATABASE :"database" TO memsystem_service;
REVOKE TEMPORARY ON DATABASE :"database" FROM PUBLIC;
SQL
