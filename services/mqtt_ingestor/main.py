"""Ingestor MQTT: consome tópicos v1, valida, persiste em lote e só então confirma (ack) as mensagens.

Garantias:
* QoS 1 + sessão persistente (MQTT v5, clean_start=False): o broker guarda mensagens enquanto o
  ingestor está fora do ar e as reentrega na reconexão.
* ack manual APÓS o commit: se o banco falhar, nada é confirmado e o lote é reprocessado.
* unicidade de event_id: reentregas e reenvios do buffer do drone não duplicam registros.
"""

from __future__ import annotations

import logging
import queue
import signal
import ssl
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import paho.mqtt.client as mqtt
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties
from sqlalchemy.exc import OperationalError

from fachadaguard.config import Settings, get_settings
from fachadaguard.db import create_schema, init_engine, session_scope
from fachadaguard.domain.models import ServiceHeartbeat, utcnow
from fachadaguard.ingestion import process_batch

log = logging.getLogger("fachadaguard.ingestor")
SERVICE = "mqtt_ingestor"


@dataclass
class Stats:
    received: int = 0
    inserted: int = 0
    duplicates: int = 0
    invalid: int = 0
    batches: int = 0
    captures: int = 0
    jobs_enqueued: int = 0
    db_retries: int = 0
    connected: bool = False
    last_latency_ms_p50: float | None = None
    started_at: str = field(default_factory=lambda: utcnow().isoformat())

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Ingestor:
    def __init__(self, settings: Settings, batch_size: int = 1000, batch_wait: float = 0.2) -> None:
        self.s = settings
        self.batch_size = batch_size
        self.batch_wait = batch_wait
        self.inbox: queue.Queue[tuple[str, bytes, int, int]] = queue.Queue()
        self.stats = Stats()
        self.stop = threading.Event()
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=settings.mqtt_client_id,
            protocol=mqtt.MQTTv5,
            manual_ack=True,
        )
        if settings.mqtt_username:
            self.client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
        if settings.mqtt_tls:
            self.client.tls_set(ca_certs=settings.mqtt_ca_file, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=15)

    # -------------------------------------------------------------- callbacks (thread do paho)
    def _on_connect(self, client: mqtt.Client, _u: Any, _f: Any, reason: Any, _p: Any) -> None:
        if reason.is_failure:
            log.error("conexão recusada pelo broker: %s", reason)
            return
        self.stats.connected = True
        p = self.s.mqtt_topic_prefix
        q = self.s.mqtt_qos
        client.subscribe(
            [(f"{p}/drones/+/telemetry", q), (f"{p}/missions/+/captures", q), (f"{p}/missions/+/status", q)]
        )
        log.info("conectado a %s:%s; inscrito em %s/#", self.s.mqtt_host, self.s.mqtt_port, p)

    def _on_disconnect(self, _c: Any, _u: Any, _f: Any, reason: Any, _p: Any) -> None:
        self.stats.connected = False
        log.warning("desconectado do broker (%s); paho fará reconexão automática", reason)

    def _on_message(self, _c: Any, _u: Any, msg: mqtt.MQTTMessage) -> None:
        self.inbox.put((msg.topic, bytes(msg.payload), msg.mid, msg.qos))

    # -------------------------------------------------------------- laço de persistência
    def _drain(self) -> list[tuple[str, bytes, int, int]]:
        batch: list[tuple[str, bytes, int, int]] = []
        deadline = time.monotonic() + self.batch_wait
        while len(batch) < self.batch_size:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                batch.append(self.inbox.get(timeout=timeout))
            except queue.Empty:
                break
        return batch

    def _persist(self, batch: list[tuple[str, bytes, int, int]]) -> None:
        delay = 0.5
        while not self.stop.is_set():
            try:
                with session_scope() as session:
                    res = process_batch(session, [(t, p) for t, p, _, _ in batch], self.s.mqtt_topic_prefix)
                break
            except OperationalError as exc:
                self.stats.db_retries += 1
                log.error(
                    "banco indisponível (%s); lote mantido sem ack, nova tentativa em %.1fs", type(exc).__name__, delay
                )
                time.sleep(delay)
                delay = min(delay * 2, 10)
        else:
            return
        for _, _, mid, qos in batch:  # ack só após commit
            if qos > 0:
                self.client.ack(mid, qos)
        st = self.stats
        st.received += res.received
        st.inserted += res.inserted
        st.duplicates += res.duplicates
        st.invalid += res.invalid
        st.captures += res.captures
        st.jobs_enqueued += res.jobs_enqueued
        st.batches += 1
        if res.latencies_ms:
            st.last_latency_ms_p50 = round(statistics.median(res.latencies_ms), 1)
        if res.invalid or res.captures or res.duplicates:
            log.info(
                "lote: %d recebidas, %d novas, %d duplicadas, %d inválidas, %d capturas",
                res.received,
                res.inserted,
                res.duplicates,
                res.invalid,
                res.captures,
            )

    def _heartbeat(self) -> None:
        try:
            with session_scope() as session:
                hb = session.get(ServiceHeartbeat, SERVICE) or ServiceHeartbeat(service=SERVICE)
                hb.last_seen, hb.stats = utcnow(), self.stats.as_dict()
                session.merge(hb)
        except OperationalError:
            log.warning("não foi possível gravar heartbeat")

    def run(self) -> None:
        props = Properties(PacketTypes.CONNECT)
        props.SessionExpiryInterval = 24 * 3600
        props.ReceiveMaximum = 2000
        self.client.connect_async(self.s.mqtt_host, self.s.mqtt_port, keepalive=30, clean_start=False, properties=props)
        self.client.loop_start()
        last_beat = 0.0
        try:
            while not self.stop.is_set():
                batch = self._drain()
                if batch:
                    self._persist(batch)
                if time.monotonic() - last_beat > 5:
                    self._heartbeat()
                    last_beat = time.monotonic()
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    init_engine(settings.database_url)
    create_schema()  # aguarda o banco e serializa a criação entre serviços
    ing = Ingestor(settings)
    signal.signal(signal.SIGTERM, lambda *_: ing.stop.set())
    signal.signal(signal.SIGINT, lambda *_: ing.stop.set())
    log.info("ingestor iniciado: %s", {k: v for k, v in settings.safe_dict().items() if k.startswith("mqtt")})
    ing.run()


if __name__ == "__main__":
    main()
