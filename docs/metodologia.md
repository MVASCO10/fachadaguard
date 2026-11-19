# Metodologia

## Desenvolvimento

Entregas incrementais em fatias verticais:

1. **Fatia vertical**: broker → ingestor → banco; upload → worker → achado → dashboard.
2. **Diferenciais**: comparação temporal, histórico e alertas, lake Parquet, benchmark, relatório.
3. **Evidências**: testes, CI, avaliação reproduzível, documentação.

Cada fase foi verificada antes da seguinte (testes automatizados, execução real da stack Docker e inspeção
visual das figuras e telas). Entre os problemas encontrados
durante a verificação e corrigidos estão a perda silenciosa de mensagens do paho acima de 65.535 pendentes e a condição de corrida na criação do esquema.

## Pipeline de IA (detalhes em `fachadaguard/vision/`)

| Etapa | Método | Saída |
|---|---|---|
| Qualidade | variância do Laplaciano, fração saturada, contraste | escore 0–1 (entra na severidade) |
| Características | patches 32 px / passo 16; Lab relativa à mediana, histograma de gradiente (9 bins), LBP uniforme, black-hat, bordas Canny, Laplaciano relativo (31 dims) | matriz por patch |
| Detecção | kNN (distância ao vizinho mais próximo) num banco de 30 mil patches normais padronizados | escore bruto |
| Calibração | quantil 0,99 dos escores em imagens normais de calibração (20%, não usadas no ajuste) | índice = 1,00 no limiar |
| Localização | média dos patches por pixel, suavização gaussiana, limiar 1,00, abertura morfológica, componentes ≥ 500 px | mapa de calor, máscara, regiões |
| Categoria candidata | regras sobre traços finos não retilíneos, escurecimento, rugosidade e elongação | categoria, escores, confiança, explicação |
| Comparação | ORB + RANSAC validado → ECC → recusa; casamento de histograma; 1 − SSIM | mapa de mudança, área alterada |
| Crescimento | máscara anterior projetada pela homografia; área atual × anterior por região | crescimento ou "achado novo" |
| Severidade | regra de pontos documentada em `fachadaguard/domain/severity.py` | nível, contribuições, notas |

## Protocolo experimental (anti-vazamento)

| Split | Semente | Uso | Ajustou algo? |
|---|---|---|---|
| `reference` (90 normais, 3 condições) | 1001 | ajuste (80%) + calibração do limiar (20%) | só o modelo |
| `dev` (60, ≈⅓ normais) | 2002 | escolha do detector, do quantil e das constantes heurísticas | sim |
| `test` (120, ≈⅓ normais) | 3003 | métricas reportadas | **não** |
| `pairs` (40) | 4004 | avaliação da comparação temporal | não |
| `demo` (6 regiões × 2 épocas) | fixas | cenário do simulador | não |

Condições de aquisição em rodízio: `nominal`, `iluminacao` (sombra projetada e ganho extremo) e
`desfoque_ruido`. Métricas: AUROC/AP de imagem, AUROC de pixel (amostrado), TPR/FPR no limiar, recall por tipo
de defeito (≥ 10% dos pixels do defeito cobertos), precisão por região, matriz de categorias candidatas,
status de alinhamento por tipo de par. Tudo em `scripts/evaluate.py` → `artifacts/evaluation/`.

## Benchmark Big Data

`scripts/benchmark.py` roda **dentro** da stack: publica N eventos únicos via MQTT (com ~2% de duplicatas
injetadas), mede o tempo até o último evento persistido no Postgres, coleta duplicatas descartadas pelo
ingestor (heartbeat), mede a latência de uma sonda em tempo real publicada durante a carga, exporta para
Parquet e cronometra consultas DuckDB. Resultados em `artifacts/benchmark/` (JSON por execução, `latest.json`
e `history.csv`).

## Como obter dados reais (próxima etapa)

1. Voos piloto em uma fachada com autorização do proprietário, operador habilitado e acesso ao espaço aéreo
   solicitado conforme as regras vigentes (ver relatório técnico, §11).
2. Plano de voo repetível (mesmos pontos, distância e ângulo), registro de horário e condições de luz.
3. Referências normais **da própria fachada** para calibrar o limiar por fachada/revestimento.
4. Rotulagem por profissional habilitado (máscaras + categoria + gravidade), com dupla anotação em amostra
   para medir concordância.
5. Validação por fachada (deixar fachadas inteiras fora do treino) e métricas por condição.

## Monitoramento e revisão do modelo

- Registrar versão do modelo, hash do conjunto de referência e parâmetros em cada execução (já feito).
- Monitorar a distribuição do índice de anomalia em regiões revisadas como normais (deriva de iluminação,
  sujeira, câmera nova) e a taxa de descartes na revisão humana.
- Retreinar/recalibrar quando a taxa de falsos positivos revisados subir ou quando mudar câmera/altitude.
