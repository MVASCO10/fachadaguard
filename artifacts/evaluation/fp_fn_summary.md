# Falsos positivos e falsos negativos — teste sintético

Gerado em 2026-11-19T01:50:11.127450+00:00 por `scripts/evaluate.py`. Dados SINTÉTICOS.

## Falsos negativos (defeito com < 10% de sobreposição): 30

| imagem | tipo | condição | pixels do defeito |
|---|---|---|---|
| test_015 | fissura | nominal | 787 |
| test_016 | umidade_infiltracao | iluminacao | 18601 |
| test_018 | umidade_infiltracao | nominal | 21827 |
| test_021 | umidade_infiltracao | nominal | 6883 |
| test_024 | umidade_infiltracao | nominal | 26119 |
| test_034 | fissura | iluminacao | 1560 |
| test_035 | umidade_infiltracao | desfoque_ruido | 11908 |
| test_044 | umidade_infiltracao | desfoque_ruido | 20903 |
| test_045 | umidade_infiltracao | nominal | 11712 |
| test_047 | fissura | desfoque_ruido | 1220 |
| test_050 | umidade_infiltracao | desfoque_ruido | 15612 |
| test_052 | umidade_infiltracao | iluminacao | 10015 |
| test_054 | umidade_infiltracao | nominal | 6956 |
| test_057 | fissura | nominal | 1217 |
| test_059 | umidade_infiltracao | desfoque_ruido | 9185 |
| test_062 | umidade_infiltracao | desfoque_ruido | 8151 |
| test_068 | umidade_infiltracao | desfoque_ruido | 9815 |
| test_075 | umidade_infiltracao | nominal | 8408 |
| test_077 | umidade_infiltracao | desfoque_ruido | 19227 |
| test_081 | umidade_infiltracao | nominal | 15253 |
| test_083 | fissura | desfoque_ruido | 1310 |
| test_086 | umidade_infiltracao | desfoque_ruido | 10153 |
| test_093 | umidade_infiltracao | nominal | 9000 |
| test_094 | fissura | iluminacao | 1340 |
| test_096 | umidade_infiltracao | nominal | 7060 |
| test_098 | umidade_infiltracao | desfoque_ruido | 12065 |
| test_106 | umidade_infiltracao | iluminacao | 11090 |
| test_107 | fissura | desfoque_ruido | 972 |
| test_110 | fissura | desfoque_ruido | 1150 |
| test_111 | umidade_infiltracao | nominal | 14878 |

## Falsos positivos (região sem defeito sintético): 57

| imagem | condição | índice de pico | área (px) | categoria candidata |
|---|---|---|---|---|
| test_001 | iluminacao | 1.43 | 4947 | indeterminado |
| test_001 | iluminacao | 1.25 | 875 | indeterminado |
| test_004 | iluminacao | 1.31 | 1060 | umidade_infiltracao |
| test_019 | iluminacao | 1.16 | 1546 | indeterminado |
| test_019 | iluminacao | 1.11 | 594 | indeterminado |
| test_024 | nominal | 1.18 | 884 | umidade_infiltracao |
| test_046 | iluminacao | 1.49 | 2335 | indeterminado |
| test_046 | iluminacao | 1.39 | 2239 | indeterminado |
| test_046 | iluminacao | 1.25 | 1746 | indeterminado |
| test_046 | iluminacao | 1.24 | 558 | indeterminado |
| test_046 | iluminacao | 1.22 | 1394 | indeterminado |
| test_046 | iluminacao | 1.21 | 1685 | indeterminado |
| test_046 | iluminacao | 1.2 | 3241 | indeterminado |
| test_046 | iluminacao | 1.2 | 1518 | indeterminado |
| test_046 | iluminacao | 1.19 | 778 | indeterminado |
| test_046 | iluminacao | 1.18 | 1535 | indeterminado |
| test_061 | iluminacao | 1.21 | 1723 | indeterminado |
| test_061 | iluminacao | 1.19 | 505 | umidade_infiltracao |
| test_071 | desfoque_ruido | 1.52 | 3003 | umidade_infiltracao |
| test_071 | desfoque_ruido | 1.5 | 3263 | umidade_infiltracao |
| test_071 | desfoque_ruido | 1.39 | 1660 | umidade_infiltracao |
| test_075 | nominal | 1.43 | 3217 | umidade_infiltracao |
| test_075 | nominal | 1.39 | 3758 | indeterminado |
| test_075 | nominal | 1.38 | 2936 | umidade_infiltracao |
| test_075 | nominal | 1.37 | 3737 | indeterminado |
| test_075 | nominal | 1.37 | 2205 | umidade_infiltracao |
| test_075 | nominal | 1.35 | 2699 | indeterminado |
| test_075 | nominal | 1.28 | 2015 | umidade_infiltracao |
| test_075 | nominal | 1.27 | 1915 | umidade_infiltracao |
| test_083 | desfoque_ruido | 1.12 | 2060 | umidade_infiltracao |
| test_083 | desfoque_ruido | 1.12 | 3077 | umidade_infiltracao |
| test_083 | desfoque_ruido | 1.1 | 1482 | umidade_infiltracao |
| test_083 | desfoque_ruido | 1.09 | 1688 | umidade_infiltracao |
| test_083 | desfoque_ruido | 1.06 | 631 | umidade_infiltracao |
| test_088 | iluminacao | 1.38 | 1330 | umidade_infiltracao |
| test_088 | iluminacao | 1.16 | 781 | indeterminado |
| test_088 | iluminacao | 1.12 | 658 | umidade_infiltracao |
| test_097 | iluminacao | 1.43 | 1971 | indeterminado |
| test_097 | iluminacao | 1.3 | 1341 | indeterminado |
| test_097 | iluminacao | 1.22 | 1501 | indeterminado |
| test_097 | iluminacao | 1.1 | 2005 | indeterminado |
| test_104 | desfoque_ruido | 1.21 | 1194 | indeterminado |
| test_104 | desfoque_ruido | 1.18 | 1380 | umidade_infiltracao |
| test_104 | desfoque_ruido | 1.18 | 1313 | indeterminado |
| test_104 | desfoque_ruido | 1.18 | 1143 | indeterminado |
| test_104 | desfoque_ruido | 1.17 | 1038 | indeterminado |
| test_104 | desfoque_ruido | 1.14 | 1069 | indeterminado |
| test_104 | desfoque_ruido | 1.14 | 1017 | indeterminado |
| test_104 | desfoque_ruido | 1.13 | 800 | indeterminado |
| test_109 | iluminacao | 4.54 | 37737 | indeterminado |
| test_110 | desfoque_ruido | 1.52 | 1381 | umidade_infiltracao |
| test_110 | desfoque_ruido | 1.43 | 4003 | umidade_infiltracao |
| test_110 | desfoque_ruido | 1.41 | 3849 | umidade_infiltracao |
| test_112 | iluminacao | 4.36 | 37190 | indeterminado |
| test_112 | iluminacao | 1.41 | 10747 | indeterminado |
| test_112 | iluminacao | 1.16 | 655 | indeterminado |
| test_112 | iluminacao | 1.07 | 792 | indeterminado |

## Agregado

Falsos negativos por (tipo, condição): fissura/desfoque_ruido=4, fissura/iluminacao=2, fissura/nominal=2, umidade_infiltracao/desfoque_ruido=9, umidade_infiltracao/iluminacao=3, umidade_infiltracao/nominal=10

Falsos positivos por condição: desfoque_ruido=19, iluminacao=29, nominal=9
