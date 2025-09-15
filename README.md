# anomfactory

Um pacote leve para **injetar anomalias sintéticas** em séries temporais **sem** depender de metadados prévios.
Ele estima internamente nível, tendência e sazonalidade e aplica anomalias selecionadas pelo usuário.

## Instalação local

```bash
pip install -e .
```

## Uso rápido

```python
import numpy as np, pandas as pd
from anomfactory import AnomalyInjector, AnomalyType

# Série de exemplo
t = np.arange(1000)
y = 100 + 0.01*t + 5*np.sin(2*np.pi*t/50) + np.random.normal(0,1, size=len(t))
s = pd.Series(y, index=pd.date_range("2024-01-01", periods=len(t), freq="min"))

inj = AnomalyInjector(random_state=123)

# 1) Espetos (spikes) positivos em 5 pontos
y1, rep1, mask1 = inj.inject(s, AnomalyType.SPIKE, severity=1.0, n_points=5, return_mask=True)

# 2) Degrau (level shift) a partir de um ponto
y2, rep2, mask2 = inj.inject(s, AnomalyType.LEVEL_SHIFT, severity=2.0, return_mask=True)

# 3) Mudança de variância em uma janela
y3, rep3, mask3 = inj.inject(s, AnomalyType.VARIANCE_CHANGE, severity=1.5, return_mask=True)
```

## Tipos de anomalia

- `SPIKE` (pontual, +)
- `DROP` (pontual, -)
- `LEVEL_SHIFT` (degrau no nível)
- `VARIANCE_CHANGE` (ruído ampliado em janela)
- `TREND_DRIFT` (mudança de inclinação)
- `SEASON_AMP_CHANGE` (sazonalidade com amplitude alterada em janela)
- `FLATLINE` (sensor preso constante em janela)
- `MISSING` (trecho com NaN)
- `STUCK_HIGH`, `STUCK_LOW` (saturação alta/baixa)
- `BLACKOUT` (zerar janela)

Cada injeção retorna também um **relatório** com índices afetados e parâmetros resolvidos.

## Design

- Sem dependências pesadas: usa `numpy`/`pandas`.
- Decomposição interna:
  - nível = mediana robusta;
  - tendência = média móvel sobre ~10% do tamanho;
  - sazonalidade = média por fase com período estimado via autocorrelação;
  - residual = observado - (nível+tendência+sazonalidade).
- Parâmetros têm defaults **robustos** derivados da própria série.

## Testes

```bash
pytest -q
```

## Licença

MIT
