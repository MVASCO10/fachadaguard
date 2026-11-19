"""Dashboard operacional do FachadaGuard (Streamlit). Consome apenas a API HTTP."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, time
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API = os.environ.get("FG_API_BASE_URL", "http://localhost:8000")
TOKEN = os.environ.get("FG_API_TOKEN", "")
HEADERS = {"X-API-Token": TOKEN} if TOKEN else {}
LEVEL_COLORS = {"critica": "#a1161b", "alta": "#d4520f", "media": "#b08900", "baixa": "#3d7a3d"}
LEVEL_ORDER = ["critica", "alta", "media", "baixa"]
METHOD_LABEL = {"orb_homography": "ORB + homografia", "ecc_affine": "ECC afim", "none": "falhou"}
REVIEW = {
    "pendente": "Pendente",
    "confirmado": "Confirmado",
    "descartado": "Descartado",
    "encaminhado_inspecao_tecnica": "Encaminhado p/ inspeção técnica",
}
CATEGORY = {
    "fissura": "Fissura (candidata)",
    "umidade_infiltracao": "Umidade/infiltração (candidata)",
    "desplacamento_aparente": "Desplacamento aparente (candidato)",
    "indeterminado": "Indeterminado",
}
DISCLAIMER = (
    "**Protótipo acadêmico de triagem.** Não é laudo, diagnóstico estrutural nem inspeção normativa. "
    "Todo achado exige revisão por profissional habilitado. Índices de anomalia não são probabilidades."
)

st.set_page_config(page_title="FachadaGuard", page_icon="🏢", layout="wide")


class ApiError(RuntimeError):
    pass


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(base_url=API, headers=HEADERS, timeout=60)


def get(path: str, **params: Any) -> Any:
    try:
        r = client().get(path, params={k: v for k, v in params.items() if v not in (None, [], "")})
    except httpx.HTTPError as exc:
        raise ApiError(f"API indisponível em {API} ({type(exc).__name__}). Verifique `docker compose ps`.") from exc
    if r.status_code >= 400:
        raise ApiError(f"{r.status_code}: {_detail(r)}")
    return r.json()


def _detail(r: httpx.Response) -> str:
    if "json" in r.headers.get("content-type", ""):
        return str(r.json().get("detail", r.text))
    return r.text[:200]


def post(path: str, **kw: Any) -> Any:
    try:
        r = client().post(path, **kw)
    except httpx.HTTPError as exc:
        raise ApiError(f"API indisponível ({type(exc).__name__})") from exc
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if "json" in r.headers.get("content-type", "") else r.text[:200]
        raise ApiError(f"{r.status_code}: {detail}")
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def image_bytes(url: str) -> bytes | None:
    try:
        r = client().get(url)
        return r.content if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def show_image(url: str | None, caption: str) -> None:
    data = image_bytes(url) if url else None
    if data:
        st.image(data, caption=caption, use_container_width=True)
    else:
        st.info(f"{caption}: imagem indisponível.")


def level_badge(level: str) -> str:
    style = (
        f"background:{LEVEL_COLORS.get(level, '#555')};color:#fff;padding:2px 9px;"
        "border-radius:10px;font-size:0.85em;font-weight:600"
    )
    return f"<span style='{style}'>{level.upper()}</span>"


def safe(fn: Any) -> Any:
    def wrapper(*a: Any, **k: Any) -> Any:
        try:
            return fn(*a, **k)
        except ApiError as exc:
            st.error(str(exc))
            return None

    return wrapper


def fmt_dt(v: Any) -> str:
    if not v:
        return "—"
    return pd.to_datetime(v, format="ISO8601", utc=True).strftime("%d/%m/%Y %H:%M UTC")


# --------------------------------------------------------------------------------------------- filtros
def sidebar_filters() -> dict[str, Any]:
    st.sidebar.title("🏢 FachadaGuard")
    st.sidebar.caption("Triagem de fachadas com drone · IoT + Big Data + IA")
    page = st.sidebar.radio(
        "Navegação",
        [
            "Visão geral",
            "Missões",
            "Inspeções",
            "Achados",
            "Comparação temporal",
            "Analítica (Big Data)",
            "Relatórios",
            "Ingestão IoT",
        ],
    )
    st.sidebar.divider()
    st.sidebar.subheader("Filtros")
    try:
        buildings = get("/api/v1/buildings")
    except ApiError as exc:
        st.sidebar.error(str(exc))
        buildings = []
    names = {b["id"]: b["name"] for b in buildings}
    building = st.sidebar.selectbox(
        "Edifício", [None, *names], format_func=lambda k: "Todos" if k is None else names[k]
    )
    facades = {f["id"]: f["name"] for b in buildings if building in (None, b["id"]) for f in b["facades"]}
    facade = st.sidebar.selectbox(
        "Fachada", [None, *facades], format_func=lambda k: "Todas" if k is None else facades[k]
    )
    period = st.sidebar.date_input("Período", value=(date(2026, 1, 1), date(2026, 12, 31)))
    severity = st.sidebar.multiselect("Severidade", LEVEL_ORDER, default=[], placeholder="Todas")
    review = st.sidebar.multiselect(
        "Status de revisão", list(REVIEW), format_func=REVIEW.get, default=[], placeholder="Todos"
    )
    st.sidebar.divider()
    st.sidebar.caption(f"API: {API}")
    d0, d1 = period if isinstance(period, tuple) and len(period) == 2 else (date(2000, 1, 1), date(2100, 1, 1))
    return {
        "page": page,
        "buildings": buildings,
        "building_id": building,
        "facade_id": facade,
        "date_from": datetime.combine(d0, time.min, UTC).isoformat(),
        "date_to": datetime.combine(d1, time.max, UTC).isoformat(),
        "severity": severity,
        "review_status": review,
    }


def findings_df(flt: dict[str, Any], **extra: Any) -> pd.DataFrame:
    rows = get(
        "/api/v1/findings",
        building_id=flt["building_id"],
        facade_id=flt["facade_id"],
        severity=flt["severity"],
        review_status=flt["review_status"],
        date_from=flt["date_from"],
        date_to=flt["date_to"],
        **extra,
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- páginas
@safe
def page_overview(flt: dict[str, Any]) -> None:
    st.title("Visão geral")
    st.warning(DISCLAIMER)
    m = get("/api/v1/metrics/summary")
    c = st.columns(6)
    c[0].metric("Missões", m["missions"])
    c[1].metric("Eventos de telemetria", f"{m['telemetry_events']:,}".replace(",", "."))
    c[2].metric("Imagens analisadas", m["captures_analyzed"], help=f"{m['captures']} capturas recebidas")
    c[3].metric("Regiões candidatas", m["findings"])
    c[4].metric("Alertas abertos", m["alerts_open"])
    c[5].metric("Mensagens rejeitadas", m["dead_letters"], help="payloads inválidos enviados à dead-letter")
    df = findings_df(flt)
    if df.empty:
        st.info("Nenhum achado para os filtros atuais. Rode a demonstração (`make demo`) ou ajuste os filtros.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Achados por severidade preliminar")
        sev = df.groupby("severity_level").size().reindex(LEVEL_ORDER, fill_value=0).reset_index(name="achados")
        st.plotly_chart(
            px.bar(
                sev,
                x="severity_level",
                y="achados",
                color="severity_level",
                color_discrete_map=LEVEL_COLORS,
                labels={"severity_level": "severidade"},
            ),
            use_container_width=True,
        )
    with right:
        st.subheader("Evolução por fachada")
        df["data"] = pd.to_datetime(df["inspected_at"], format="ISO8601", utc=True).dt.date.astype(str)
        evo = df.groupby(["facade_name", "data"]).agg(achados=("id", "count"), area=("area_ratio", "sum")).reset_index()
        evo["area_%"] = (evo["area"] * 100).round(2)
        st.plotly_chart(
            px.bar(
                evo,
                x="data",
                y="area_%",
                color="facade_name",
                barmode="group",
                hover_data=["achados"],
                labels={"area_%": "área anômala somada (%)", "facade_name": "fachada"},
            ).update_xaxes(type="category"),
            use_container_width=True,
        )
    st.subheader("Alertas priorizados")
    alerts = get("/api/v1/alerts")
    if not alerts:
        st.success("Nenhum alerta.")
    for a in alerts[:8]:
        st.markdown(
            f"{level_badge(a['level'])} &nbsp; `{a['finding_id']}` · {a['status']} — {a['reason']}",
            unsafe_allow_html=True,
        )


@safe
def page_missions(flt: dict[str, Any]) -> None:
    st.title("Missões")
    missions = get("/api/v1/missions", building_id=flt["building_id"])
    if not missions:
        st.info("Nenhuma missão cadastrada. Execute `make simulate-a`.")
        return
    tbl = pd.DataFrame(missions)[
        ["id", "name", "status", "is_simulated", "planned_for", "telemetry_events", "captures", "open_alerts"]
    ]
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    mid = st.selectbox(
        "Detalhar missão",
        [m["id"] for m in missions],
        format_func=lambda i: next(m["name"] for m in missions if m["id"] == i),
    )
    with st.spinner("Carregando missão..."):
        detail = get(f"/api/v1/missions/{mid}")
        tel = pd.DataFrame(get(f"/api/v1/missions/{mid}/telemetry", limit=5000))
    if detail["is_simulated"]:
        st.caption("🧪 Missão **SIMULADA**: telemetria e imagens geradas pelo simulador, dados sintéticos.")
    if tel.empty:
        st.info("Sem telemetria para esta missão.")
    else:
        c = st.columns(4)
        c[0].metric("Eventos", len(tel))
        c[1].metric("Bateria mínima", f"{tel['battery_pct'].min():.1f}%")
        c[2].metric("Sinal mínimo", f"{tel['signal_quality'].min():.0f}/100")
        c[3].metric("Altitude máx.", f"{tel['altitude_m'].max():.1f} m")
        a, b = st.columns(2)
        with a:
            st.plotly_chart(
                px.scatter(
                    tel,
                    x="longitude",
                    y="latitude",
                    color="mission_state",
                    title="Trajetória (coordenadas WGS84, sem mapa base)",
                    height=380,
                ),
                use_container_width=True,
            )
        with b:
            tel["t"] = pd.to_datetime(tel["captured_at"], format="ISO8601", utc=True)
            long = tel.melt(id_vars="t", value_vars=["battery_pct", "signal_quality", "altitude_m"])
            st.plotly_chart(
                px.line(long, x="t", y="value", color="variable", title="Bateria, sinal e altitude", height=380),
                use_container_width=True,
            )
    st.subheader("Inspeções da missão")
    for insp in detail["inspections"]:
        reg = insp["region"] or {}
        runs = [r for c in insp["captures"] for r in c["runs"]]
        last = runs[-1] if runs else None
        st.markdown(
            f"**{reg.get('code', '?')}** — {reg.get('description', '')} · {fmt_dt(insp['inspected_at'])} · "
            f"análise: `{last['status'] if last else 'aguardando'}`"
            + (f" · índice máx. {last['image_index']:.2f}" if last and last.get("image_index") else "")
        )
    if detail["captures_awaiting_metadata"]:
        st.warning(f"{len(detail['captures_awaiting_metadata'])} imagens aguardando metadados MQTT.")


@safe
def page_inspections(flt: dict[str, Any]) -> None:
    st.title("Inspeções")
    rows = get("/api/v1/inspections", building_id=flt["building_id"], facade_id=flt["facade_id"])
    if not rows:
        st.info("Nenhuma inspeção. Execute a simulação ou envie uma imagem abaixo.")
    else:
        for row in rows:
            an = row["analysis"]
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 1, 1.2])
                with c1:
                    show_image(
                        row["capture"]["image_url"] if row["capture"] else None,
                        f"Original · {row['region_code']} · {fmt_dt(row['inspected_at'])}",
                    )
                with c2:
                    show_image(an["overlay_url"] if an else None, "Mapa de calor do índice de anomalia")
                with c3:
                    st.markdown(f"**{row['facade_name']} · {row['region_code']}**  \n{row['mission_name']}")
                    if row["is_simulated"]:
                        st.caption("🧪 dados sintéticos")
                    if an:
                        st.metric("Índice máximo (1,00 = limiar)", f"{an['image_index']:.2f}")
                        st.write(
                            f"Área acima do limiar: **{an['anomalous_area_ratio']:.2%}** · "
                            f"achados: **{an['n_findings']}**"
                        )
                        q = an["quality"] or {}
                        st.write(
                            f"Qualidade da evidência: **{q.get('quality_score', 0):.2f}** "
                            f"(nitidez {q.get('sharpness_score', 0):.2f}, exposição {q.get('exposure_score', 0):.2f})"
                        )
                        st.caption(f"Modelo `{an['model_version']}` · {an['duration_ms']:.0f} ms")
                    else:
                        st.info("Análise na fila ou captura incompleta.")
    with st.expander("➕ Enviar imagem manualmente para uma região"):
        missions = get("/api/v1/missions", building_id=flt["building_id"])
        regions = {
            r["id"]: f"{f['name']} · {r['code']}" for b in flt["buildings"] for f in b["facades"] for r in f["regions"]
        }
        if not missions or not regions:
            st.info("Cadastre uma missão e regiões primeiro.")
            return
        with st.form("upload"):
            mid = st.selectbox(
                "Missão",
                [m["id"] for m in missions],
                format_func=lambda i: next(m["name"] for m in missions if m["id"] == i),
            )
            rid = st.selectbox("Região", list(regions), format_func=regions.get)
            up = st.file_uploader("Imagem PNG/JPEG (até 15 MB)", type=["png", "jpg", "jpeg"])
            if st.form_submit_button("Enviar e analisar") and up is not None:
                res = post(
                    f"/api/v1/missions/{mid}/images",
                    files={"file": (up.name, up.getvalue(), up.type)},
                    data={"region_id": rid},
                )
                st.success(
                    f"Captura {res['capture']['id']} enviada; análise "
                    f"{'enfileirada' if res['analysis'] else 'não enfileirada'}."
                )


@safe
def page_findings(flt: dict[str, Any]) -> None:
    st.title("Achados e revisão humana")
    df = findings_df(flt)
    if df.empty:
        st.info("Nenhum achado para os filtros atuais.")
        return
    view = df[
        [
            "id",
            "facade_name",
            "region_code",
            "inspected_at",
            "candidate_category",
            "category_confidence",
            "severity_level",
            "severity_points",
            "area_ratio",
            "peak_index",
            "review_status",
        ]
    ].copy()
    view["candidate_category"] = view["candidate_category"].map(CATEGORY)
    view["review_status"] = view["review_status"].map(REVIEW)
    view["inspected_at"] = view["inspected_at"].map(fmt_dt)
    view["area_ratio"] = (view["area_ratio"] * 100).round(2)
    view = view.rename(
        columns={
            "facade_name": "fachada",
            "region_code": "região",
            "inspected_at": "inspeção",
            "candidate_category": "categoria candidata",
            "category_confidence": "confiança",
            "severity_level": "severidade",
            "severity_points": "pontos",
            "area_ratio": "área (%)",
            "peak_index": "índice de pico",
            "review_status": "revisão",
        }
    )
    st.dataframe(
        view,
        hide_index=True,
        use_container_width=True,
        column_config={"índice de pico": st.column_config.NumberColumn(format="%.2f")},
    )
    fid = st.selectbox(
        "Detalhar achado",
        df["id"].tolist(),
        format_func=lambda i: (
            f"{i} · {df.set_index('id').loc[i, 'region_code']} · {df.set_index('id').loc[i, 'severity_level']}"
        ),
    )
    f = get(f"/api/v1/findings/{fid}")
    st.markdown(
        f"### {level_badge(f['severity_level'])} &nbsp; {CATEGORY.get(f['candidate_category'])} "
        f"· {f['severity_points']:.0f}/100 pts",
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    with a:
        show_image(
            f["capture"]["image_url"],
            f"Original (caixa x={f['bbox']['x']}, y={f['bbox']['y']}, {f['bbox']['w']}×{f['bbox']['h']} px)",
        )
    with b:
        show_image(f["analysis"]["overlay_url"], "Mapa de calor + contorno acima do limiar")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Detecção de anomalia")
        st.write(f"Índice de pico: **{f['peak_index']:.2f}** (limiar = 1,00; não é probabilidade)")
        st.write(f"Área relativa: **{f['area_ratio']:.2%}**")
        st.caption(
            f"Modelo `{f['analysis']['model_version']}` · "
            f"referência `{(f['analysis']['reference_set_hash'] or '')[:12]}`"
        )
    with c2:
        st.markdown("#### 2. Categoria candidata (heurística)")
        st.write(f"**{CATEGORY.get(f['candidate_category'])}** · confiança **{f['category_confidence']}**")
        st.caption(f["category_explanation"])
        st.json(f["category_scores"], expanded=False)
    with c3:
        st.markdown("#### 3. Por que foi priorizado")
        sev = f["severity_explanation"]
        st.dataframe(
            pd.DataFrame(sev["contributions"])[["fator", "pontos", "explicacao"]],
            hide_index=True,
            use_container_width=True,
        )
        for n in sev.get("notes", []):
            st.caption(f"• {n}")
    st.markdown("#### 4. Decisão humana")
    with st.form("review"):
        status = st.radio(
            "Status",
            list(REVIEW),
            format_func=REVIEW.get,
            horizontal=True,
            index=list(REVIEW).index(f["review_status"]),
        )
        note = st.text_area("Observação", placeholder="Ex.: confirmado visualmente; solicitar vistoria presencial.")
        reviewer = st.text_input("Revisor", value="revisor-demo")
        if st.form_submit_button("Registrar revisão"):
            post(f"/api/v1/findings/{fid}/review", json={"status": status, "note": note or None, "reviewer": reviewer})
            st.success("Revisão registrada. Nenhuma ação física é disparada automaticamente.")
            st.rerun()
    if f["reviews"]:
        st.dataframe(pd.DataFrame(f["reviews"]), hide_index=True, use_container_width=True)


@safe
def page_compare(flt: dict[str, Any]) -> None:
    st.title("Comparação temporal")
    rows = get("/api/v1/inspections", building_id=flt["building_id"], facade_id=flt["facade_id"])
    if len(rows) < 2:
        st.info("São necessárias ao menos duas inspeções.")
        return
    label = {r["id"]: f"{r['facade_name']} · {r['region_code']} · {fmt_dt(r['inspected_at'])}" for r in rows}
    regions = sorted({r["region_code"] for r in rows})
    reg = st.selectbox("Região", regions)
    same = [r["id"] for r in rows if r["region_code"] == reg]
    c1, c2 = st.columns(2)
    before = c1.selectbox("Inspeção anterior", list(label), index=list(label).index(same[0]), format_func=label.get)
    after = c2.selectbox("Inspeção posterior", list(label), index=list(label).index(same[-1]), format_func=label.get)
    rb = next(r for r in rows if r["id"] == before)
    ra = next(r for r in rows if r["id"] == after)
    force = False
    if rb["region_id"] != ra["region_id"]:
        st.error("As inspeções são de **regiões diferentes**: a comparação automática não representa evolução.")
        force = st.checkbox("Comparar mesmo assim (apenas exploratório)")
    if st.button("Comparar", type="primary", disabled=before == after):
        with st.spinner("Alinhando (ORB + homografia / ECC) e calculando SSIM..."):
            res = post(
                "/api/v1/comparisons",
                json={"before_inspection_id": before, "after_inspection_id": after, "force": force},
            )
        st.session_state["cmp"] = res
    res = st.session_state.get("cmp")
    if not res:
        cmps = get("/api/v1/comparisons")
        if cmps:
            st.caption("Comparações automáticas mais recentes (geradas pelo worker):")
            st.dataframe(
                pd.DataFrame(cmps)[
                    [
                        "id",
                        "status",
                        "interval_days",
                        "ssim",
                        "changed_area_ratio",
                        "anomalous_area_before",
                        "anomalous_area_after",
                    ]
                ],
                hide_index=True,
            )
        return
    for w in res["warnings"]:
        st.warning(w)
    c = st.columns(5)
    c[0].metric("Intervalo", f"{res['interval_days']:.0f} dias")
    c[1].metric(
        "Alinhamento",
        METHOD_LABEL.get(res["alignment"]["method"], res["alignment"]["method"]),
        help=res["alignment"].get("reason") or "validado",
    )
    c[2].metric("SSIM médio", f"{res['ssim']:.3f}" if res["ssim"] is not None else "—")
    c[3].metric("Área alterada", f"{res['changed_area_ratio']:.2%}" if res["changed_area_ratio"] is not None else "—")
    if res["anomalous_area_before"] is not None and res["anomalous_area_after"] is not None:
        c[4].metric(
            "Área anômala",
            f"{res['anomalous_area_after']:.2%}",
            delta=f"{(res['anomalous_area_after'] - res['anomalous_area_before']) * 100:+.2f} p.p.",
        )
    i1, i2, i3 = st.columns(3)
    with i1:
        show_image(res["image_before_url"], "Antes (original)")
    with i2:
        show_image(res["image_after_url"], "Depois (original)")
    with i3:
        show_image(res["overlay_url"], "Mapa de mudança (1 − SSIM) após alinhamento")
    with st.expander("Métricas de alinhamento"):
        st.json(res["alignment"])


@safe
def page_analytics(_: dict[str, Any]) -> None:
    st.title("Analítica histórica (Parquet + DuckDB)")
    st.caption(
        "Caminho analítico: o worker exporta periodicamente Postgres → Parquet particionado "
        "(dt / mission_id); as consultas abaixo rodam no DuckDB sobre o lake."
    )
    if st.button("Exportar agora para o lake"):
        with st.spinner("Exportando..."):
            st.json(post("/api/v1/analytics/export"))
    stats = get("/api/v1/analytics/lake-stats")
    c = st.columns(3)
    c[0].metric("Arquivos Parquet", stats["parquet_files"])
    c[1].metric("Tamanho do lake", f"{stats['bytes'] / 1e6:.1f} MB")
    c[2].metric("Último id exportado", stats["telemetry_last_id"])
    tm = pd.DataFrame(get("/api/v1/analytics/telemetry-by-mission"))
    if tm.empty:
        st.info("Lake vazio. Clique em 'Exportar agora' ou aguarde o worker.")
        return
    st.subheader("Telemetria por missão")
    st.dataframe(tm, hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    with a:
        dist = pd.DataFrame(get("/api/v1/analytics/telemetry-distribution"))
        g = dist.groupby("faixa_bateria", as_index=False)["eventos"].sum()
        st.plotly_chart(
            px.bar(g, x="faixa_bateria", y="eventos", title="Distribuição de bateria (eventos)"),
            use_container_width=True,
        )
    with b:
        g = dist.groupby("faixa_sinal", as_index=False)["eventos"].sum()
        st.plotly_chart(
            px.bar(g, x="faixa_sinal", y="eventos", title="Distribuição da qualidade de sinal"),
            use_container_width=True,
        )
    epd = pd.DataFrame(get("/api/v1/analytics/events-per-day"))
    st.plotly_chart(
        px.bar(epd, x="dt", y="eventos", hover_data=["missoes", "drones"], title="Eventos por dia"),
        use_container_width=True,
    )
    an = pd.DataFrame(get("/api/v1/analytics/anomalies-by-facade"))
    if not an.empty:
        st.subheader("Evolução de anomalias por fachada e região")
        an["data_inspecao"] = an["data_inspecao"].astype(str)
        st.plotly_chart(
            px.line(
                an,
                x="data_inspecao",
                y="area_total_pct",
                color="region_code",
                markers=True,
                facet_col="facade_name",
                labels={"area_total_pct": "área anômala (%)"},
            ),
            use_container_width=True,
        )
        st.dataframe(an, hide_index=True, use_container_width=True)


@safe
def page_reports(flt: dict[str, Any]) -> None:
    st.title("Relatórios preliminares")
    missions = get("/api/v1/missions", building_id=flt["building_id"])
    if not missions:
        st.info("Nenhuma missão.")
        return
    mid = st.selectbox(
        "Missão", [m["id"] for m in missions], format_func=lambda i: next(m["name"] for m in missions if m["id"] == i)
    )
    if st.button("Gerar relatório", type="primary"):
        with st.spinner("Gerando..."):
            rep = post(f"/api/v1/missions/{mid}/reports")
        st.success(f"Relatório {rep['id']} gerado ({rep['generator']}).")
    reports = get(f"/api/v1/missions/{mid}")["reports"]
    if not reports:
        st.info("Nenhum relatório gerado para esta missão.")
        return
    rep = reports[0]
    html = client().get(rep["url"]).text
    st.download_button("⬇ Baixar HTML", html, file_name=f"relatorio_{mid}.html", mime="text/html")
    st.caption(
        f"SHA-256 {rep['sha256'][:16]}… · {fmt_dt(rep['generated_at'])} · para PDF use 'Imprimir → Salvar como PDF'."
    )
    st.components.v1.html(html, height=900, scrolling=True)


@safe
def page_ingestion(_: dict[str, Any]) -> None:
    st.title("Estado da ingestão IoT")
    s = get("/api/v1/ingestion/status")
    if not s["services"]:
        st.warning("Nenhum heartbeat recebido dos serviços (ingestor/worker).")
    for svc in s["services"]:
        icon = "🟢" if svc["alive"] else "🔴"
        with st.container(border=True):
            st.markdown(f"{icon} **{svc['service']}** · último sinal {fmt_dt(svc['last_seen'])}")
            st.json(svc["stats"], expanded=False)
    st.subheader("Eventos recentes")
    ev = pd.DataFrame(s["recent_events"])
    # if/else explícito: um ternário solto seria capturado pela "magic" do Streamlit (StreamlitAPIException)
    if ev.empty:
        st.info("Sem eventos.")
    else:
        st.dataframe(ev, hide_index=True, use_container_width=True)
    st.subheader("Mensagens rejeitadas (dead-letter)")
    dl = pd.DataFrame(s["dead_letters"])
    if dl.empty:
        st.success("Nenhuma.")
    else:
        st.dataframe(dl, hide_index=True, use_container_width=True)
    if st.button("Atualizar"):
        st.rerun()


PAGES = {
    "Visão geral": page_overview,
    "Missões": page_missions,
    "Inspeções": page_inspections,
    "Achados": page_findings,
    "Comparação temporal": page_compare,
    "Analítica (Big Data)": page_analytics,
    "Relatórios": page_reports,
    "Ingestão IoT": page_ingestion,
}

filters = sidebar_filters()
PAGES[filters["page"]](filters)
st.sidebar.caption("Protótipo acadêmico — não é laudo técnico.")
