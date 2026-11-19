"""Regra de severidade PRELIMINAR, transparente e auditável (0–100 pontos).

Componentes (somados, depois ponderados pela qualidade da evidência):

==========================  ======  ===========================================================
componente                  máx.    cálculo
==========================  ======  ===========================================================
intensidade da anomalia      35     35 · clip((índice_pico − 1) / 2, 0, 1)   (índice 1 = limiar)
área relativa                25     25 · clip(área_relativa / 0,04, 0, 1)
crescimento temporal         25     25 · clip(crescimento / 1,0, 0, 1); região nova = 15
categoria candidata          15     desplacamento 15 · fissura 8 · umidade 5 · indeterminado 3,
                                    multiplicado por 0,5 se a confiança heurística for baixa
==========================  ======  ===========================================================

Fator de evidência = 0,6 + 0,4 · qualidade (0–1). Com qualidade < 0,4 o nível fica limitado a
"media" e o achado é marcado como evidência insuficiente.

Níveis: baixa < 30 ≤ média < 55 ≤ alta < 75 ≤ crítica. Alta e crítica geram alerta para revisão
humana; nenhuma ação física é disparada automaticamente. O índice NÃO é probabilidade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RULE_VERSION = "severidade-1.0"
LEVELS = (("baixa", 0.0), ("media", 30.0), ("alta", 55.0), ("critica", 75.0))
CATEGORY_POINTS = {"desplacamento_aparente": 15.0, "fissura": 8.0, "umidade_infiltracao": 5.0, "indeterminado": 3.0}
ALERT_LEVELS = {"alta", "critica"}


@dataclass(frozen=True)
class Growth:
    has_history: bool
    is_new: bool = False
    growth_ratio: float | None = None  # (área_atual − área_anterior) / área_anterior


@dataclass
class SeverityResult:
    points: float
    level: str
    contributions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def should_alert(self) -> bool:
        return self.level in ALERT_LEVELS

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_version": RULE_VERSION,
            "points": self.points,
            "level": self.level,
            "contributions": self.contributions,
            "notes": self.notes,
        }


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def level_for(points: float) -> str:
    level = LEVELS[0][0]
    for name, lower in LEVELS:
        if points >= lower:
            level = name
    return level


def assess(
    peak_index: float,
    area_ratio: float,
    growth: Growth,
    quality_score: float,
    category: str,
    category_confidence: str,
) -> SeverityResult:
    contrib: list[dict[str, Any]] = []
    notes: list[str] = []

    p_int = 35 * _clip((peak_index - 1.0) / 2.0)
    contrib.append(
        {
            "fator": "intensidade_anomalia",
            "valor": round(peak_index, 3),
            "pontos": round(p_int, 1),
            "explicacao": f"índice de pico {peak_index:.2f} (1,00 = limiar calibrado em referências normais)",
        }
    )
    p_area = 25 * _clip(area_ratio / 0.04)
    contrib.append(
        {
            "fator": "area_relativa",
            "valor": round(area_ratio, 4),
            "pontos": round(p_area, 1),
            "explicacao": f"região ocupa {area_ratio:.2%} da imagem (25 pts a partir de 4%)",
        }
    )
    if not growth.has_history:
        p_growth = 0.0
        expl = "sem inspeção anterior comparável desta região"
        notes.append("Sem histórico: crescimento não avaliado.")
    elif growth.is_new:
        p_growth = 15.0
        expl = "região alterada ausente na inspeção anterior (achado novo)"
    else:
        g = growth.growth_ratio or 0.0
        p_growth = 25 * _clip(g / 1.0)
        expl = f"variação de área de {g:+.0%} desde a inspeção anterior"
    contrib.append(
        {
            "fator": "crescimento_temporal",
            "valor": growth.growth_ratio,
            "pontos": round(p_growth, 1),
            "explicacao": expl,
        }
    )
    p_cat = CATEGORY_POINTS.get(category, 3.0) * (0.5 if category_confidence == "baixa" else 1.0)
    contrib.append(
        {
            "fator": "categoria_candidata",
            "valor": category,
            "pontos": round(p_cat, 1),
            "explicacao": f"categoria candidata '{category}' (confiança heurística {category_confidence})",
        }
    )

    raw = p_int + p_area + p_growth + p_cat
    factor = 0.6 + 0.4 * _clip(quality_score)
    points = round(raw * factor, 1)
    contrib.append(
        {
            "fator": "qualidade_evidencia",
            "valor": round(quality_score, 3),
            "pontos": None,
            "explicacao": f"multiplicador {factor:.2f} sobre {raw:.1f} pontos brutos",
        }
    )
    level = level_for(points)
    if quality_score < 0.4:
        notes.append("Evidência de baixa qualidade: nível limitado a 'media'; recomenda-se nova captura.")
        if level in ALERT_LEVELS:
            level = "media"
    return SeverityResult(points=points, level=level, contributions=contrib, notes=notes)


def explain(result: SeverityResult) -> str:
    top = sorted((c for c in result.contributions if c["pontos"]), key=lambda c: c["pontos"], reverse=True)[:3]
    parts = "; ".join(f"{c['explicacao']} (+{c['pontos']:.0f})" for c in top)
    return f"Severidade preliminar {result.level.upper()} ({result.points:.0f}/100). Principais fatores: {parts}."
