# Dataset card — fachadas sintéticas FachadaGuard

| Campo | Valor |
|---|---|
| Nome / versão | `synthfacade-1.2` (`fachadaguard/vision/synthetic.py`) |
| Natureza | **100% sintético**, gerado proceduralmente por código; nenhuma imagem real |
| Licença | a mesma do código (MIT); sem direitos de terceiros |
| Finalidade | validar o funcionamento técnico do pipeline (ingestão, IA, comparação, severidade) e expor modos de falha |
| **Não serve para** | estimar desempenho em fachadas reais, comparar com a literatura, decidir intervenções |
| Geração | `python scripts/generate_demo_dataset.py` → `data/synthetic/` com `manifest.json` e SHA-256 por arquivo; `--verify` confere integridade |
| Resolução | 640 × 480 px, BGR 8 bits; máscaras PNG com rótulos 0–3 |

## Conteúdo

- **Fachada base**: painéis/placas com variação de tom, juntas, ruído fino e de baixa frequência, gradiente de
  iluminação, faixa de janelas opcional; 6 paletas (bege, cinza, terracota, azul ACM, branco, grafite).
- **Defeitos** (com estágio 0–1 para simular evolução):
  - `1` fissura — caminho aleatório escuro de 1–2 px com ramificações;
  - `2` mancha compatível com umidade — manchas gaussianas que "escorrem", escurecimento e matiz marrom/verde;
  - `3` desplacamento aparente — polígono irregular com substrato cinza rugoso, borda e sombra.
- **Condições de aquisição**: ganho/gama, sombra projetada, desfoque, ruído e mudança de ponto de vista
  (rotação, translação, escala, perspectiva).

## Splits

Ver [`metodologia.md`](metodologia.md#protocolo-experimental-anti-vazamento). Sementes fixas; os splits são
disjuntos (teste automatizado `tests/unit/test_dataset.py`).

## Cenário de demonstração (edifício fictício "Shopping Atlântico")

| Região | Época A (12/05/2026) | Época B (10/11/2026) | Propósito |
|---|---|---|---|
| N-01 | fissura curta (45%) | fissura longa + ramos | crescimento |
| N-02 | normal | normal | controle |
| N-03 | mancha pequena (35%) | mancha maior | mancha em expansão (caso difícil) |
| L-01 | fissura fina (30%) | **desplacamento novo** + fissura | achado novo |
| L-02 | fissura | mesma fissura | estabilidade |
| L-03 | normal | normal **com sombra forte** | falso positivo de iluminação |

A época B tem outro ponto de vista (translação, rotação de 1,4°, escala 1,02) e ganho 0,93, para exigir
alinhamento. Coordenadas do edifício são fictícias.

## Limitações e vieses

- Defeitos e texturas são simplificações; fissuras reais têm contraste, largura e contexto muito mais
  variados, e manchas reais se confundem com sujeira, reflexos e sombras.
- Não há pessoas, placas, vegetação, chuva, reflexo especular de vidro, ângulos oblíquos fortes nem compressão
  JPEG de drones reais.
- As constantes das heurísticas foram ajustadas neste domínio sintético e precisarão de recalibração.
- As imagens de demo e de teste vêm do mesmo gerador que as referências: o domínio é "fácil" e otimista.
