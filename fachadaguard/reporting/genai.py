"""Adaptador OPCIONAL de IA generativa para redigir o resumo executivo do relatório.

Desligado por padrão (FG_GENAI_ENABLED=false). Só é usado quando habilitado, com chave configurada
e o pacote ``anthropic`` instalado (``pip install -e '.[genai]'``). O modelo recebe APENAS os fatos
estruturados já calculados; a saída é validada e descartada se:

* citar números que não constam dos fatos;
* usar termos proibidos (laudo, diagnóstico, norma, NBR, certificação, garantia, etc.);
* exceder o tamanho máximo.

Em qualquer falha, o relatório usa o resumo determinístico do template.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fachadaguard.config import Settings

log = logging.getLogger(__name__)
FORBIDDEN = re.compile(
    r"\b(laudo|diagn[oó]stic\w*|nbr|norma\w*|certific\w*|garant\w*|aprovad\w*|seguro para|sem risco)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
MAX_CHARS = 1200

SYSTEM_PROMPT = (
    "Você redige o resumo executivo de um relatório PRELIMINAR de triagem visual de fachada gerado por um "
    "protótipo acadêmico. Escreva em português do Brasil, 3 a 5 frases, tom técnico e cauteloso. Use "
    "somente os fatos JSON fornecidos; não acrescente medições, causas, normas, diagnósticos ou "
    "recomendações estruturais. Toda categoria é 'candidata' e exige revisão por profissional habilitado. "
    "Não use as palavras laudo, diagnóstico, norma ou certificação."
)


def _allowed_numbers(facts: Any) -> set[str]:
    text = json.dumps(facts, ensure_ascii=False)
    values = set()
    for raw in NUMBER.findall(text):
        values.add(raw.replace(",", "."))
        try:
            f = float(raw.replace(",", "."))
            values.update({f"{f:g}", f"{f:.0f}", f"{f:.1f}", f"{f:.2f}", f"{f * 100:.0f}", f"{f * 100:.1f}"})
        except ValueError:  # pragma: no cover
            pass
    return values


def validate_summary(text: str, facts: dict[str, Any]) -> list[str]:
    problems = []
    if not text.strip():
        problems.append("texto vazio")
    if len(text) > MAX_CHARS:
        problems.append("texto longo demais")
    if FORBIDDEN.search(text):
        problems.append(f"termo proibido: {FORBIDDEN.search(text).group(0)}")  # type: ignore[union-attr]
    allowed = _allowed_numbers(facts)
    unknown = [n for n in NUMBER.findall(text) if n.replace(",", ".") not in allowed]
    if unknown:
        problems.append(f"números sem lastro nos fatos: {sorted(set(unknown))[:5]}")
    return problems


def generate_summary(settings: Settings, facts: dict[str, Any]) -> tuple[str | None, str]:
    """Retorna (texto_validado | None, motivo)."""
    if not settings.genai_enabled:
        return None, "IA generativa desabilitada por configuração"
    if not settings.anthropic_api_key:
        return None, "FG_ANTHROPIC_API_KEY ausente"
    try:
        import anthropic  # dependência opcional
    except ImportError:
        return None, "pacote 'anthropic' não instalado"
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=60.0, max_retries=2)
    try:
        response = client.beta.messages.create(
            model=settings.genai_model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": "Fatos estruturados:\n" + json.dumps(facts, ensure_ascii=False)}],
        )
    except anthropic.APIConnectionError:
        return None, "falha de conexão com a API"
    except anthropic.RateLimitError:
        return None, "limite de requisições atingido"
    except anthropic.APIStatusError as exc:
        return None, f"erro da API ({exc.status_code})"
    if response.stop_reason == "refusal":
        return None, "modelo recusou a solicitação"
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    problems = validate_summary(text, facts)
    if problems:
        log.warning("resumo generativo descartado: %s", "; ".join(problems))
        return None, "saída descartada na validação: " + "; ".join(problems)
    return text, f"redigido por {response.model} e validado contra os fatos"
