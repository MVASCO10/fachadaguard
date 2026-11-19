#!/bin/sh
# Gera o arquivo de senhas (hash PBKDF2) a partir de variáveis de ambiente; nada é versionado.
set -eu
: "${MQTT_DRONE_PASSWORD:?defina MQTT_DRONE_PASSWORD}"
: "${MQTT_INGESTOR_PASSWORD:?defina MQTT_INGESTOR_PASSWORD}"
mkdir -p /mosquitto/config
cp /fg/mosquitto.conf /mosquitto/config/mosquitto.conf
cp /fg/acl /mosquitto/config/acl
rm -f /mosquitto/config/passwd
touch /mosquitto/config/passwd
mosquitto_passwd -b /mosquitto/config/passwd fg_drone "$MQTT_DRONE_PASSWORD"
mosquitto_passwd -b /mosquitto/config/passwd fg_ingestor "$MQTT_INGESTOR_PASSWORD"
chown mosquitto:mosquitto /mosquitto/config/passwd /mosquitto/config/acl
chmod 0700 /mosquitto/config/passwd /mosquitto/config/acl
exec mosquitto -c /mosquitto/config/mosquitto.conf
