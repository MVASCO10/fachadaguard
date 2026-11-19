# data/

Diretório de dados locais (não versionado, exceto este arquivo).

| Caminho | Conteúdo | Como gerar |
|---|---|---|
| `synthetic/` | dataset **sintético** (referência, dev, teste, demo) + `manifest.json` com SHA-256 | `make dataset` |
| `store/` | imagens recebidas, mapas de calor, comparações e relatórios (quando a API roda fora do Docker) | automático |
| `lake/` | Parquet particionado + `_state.json` (marca d'água) | `make export` |
| `sim_buffer/` | buffer local do drone simulado (SQLite) | automático |

Na stack Docker esses dados ficam nos volumes `fg_data` e `fg_artifacts`.

Nenhuma imagem real ou de terceiros é usada. Para incluir imagens públicas reais, registre fonte, licença,
finalidade e limitações em `docs/dataset_card.md`, não as versione no Git e forneça script de download com
checksum. Ficha completa do dataset: [`docs/dataset_card.md`](../docs/dataset_card.md).
