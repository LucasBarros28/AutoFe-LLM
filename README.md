# AutoFE — Automated Feature Engineering com LLMs

Ferramenta de engenharia automática de atributos (AutoFE) para dados tabulares. O projeto detecta oportunidades de transformação, gera novas features, seleciona as mais relevantes e mede o impacto em modelos de Machine Learning. Uma LLM (Groq ou Gemini) pode sugerir transformações, mas **nunca executa código**: as sugestões passam por uma lista de operações permitidas e são executadas pelo motor tradicional do projeto.

## Objetivos

* Identificar oportunidades de engenharia de atributos.
* Gerar automaticamente novas features.
* Avaliar o impacto das transformações em modelos de ML.
* Comparar AutoFE tradicional com AutoFE assistido por LLM.
* Manter uma estrutura reutilizável para diferentes datasets.

## Arquitetura

```
Dataset ──► OpportunityDetector ──► FeatureGenerator ──► FeatureSelector ──► Modelo
                    ▲                                                        │
                    │                                                        ▼
          LLM (Groq / Gemini) ── sugestões validadas ──┘               ModelEvaluator
          (recebe só estatísticas das colunas)                         (CV 5 folds, auditoria)
```

|Componente|Arquivo|Função|
|-|-|-|
|Data Loader|`src/data\_loader.py`|Carregamento e preparação dos datasets.|
|Opportunity Detector|`src/analysis/opportunity\_detector.py`|Detecta assimetria, ausentes, outliers, colunas constantes e candidatas a interação.|
|Feature Generator|`src/analysis/feature\_generator.py`|Executa as transformações `log1p`, `missing\_indicator` e `interaction\_product`.|
|AutoFE Transformer|`src/analysis/autofe\_transformer.py`|Integra tudo como transformer do Scikit-learn; orquestra a LLM, cache e circuit breaker.|
|Feature Selector|`src/analysis/feature\_selector.py`|Seleciona as `k` melhores features.|
|LLM Clients|`src/llm/groq\_client.py`, `src/llm/gemini\_client.py`|Chamam a LLM, aplicam retry e validam a resposta contra a lista de operações permitidas.|
|Model Evaluator|`src/evaluation/model\_evaluator.py`|Treina, avalia com validação cruzada e registra a auditoria por fold.|
|Relatório|`src/report.py`|Consolida métricas, features e sugestões da LLM em `results/report/`.|

### Segurança da integração com LLM

* A LLM recebe apenas metadados das colunas (tipo, % de ausentes, média, desvio, mínimo, máximo, assimetria), não as linhas do dataset.
* Só são aceitas as operações `log1p`, `missing\_indicator` e `interaction\_product`; colunas inexistentes ou inválidas são rejeitadas.
* Máximo de 5 sugestões por consulta; respostas ficam em cache local (`.cache/`).
* Circuit breaker interrompe novas chamadas quando a cota da API se esgota, e o cenário é marcado como **incompleto**.

## Cenários experimentais

|Cenário|AutoFE|LLM|Interações|Seleção|
|-|:-:|:-:|:-:|:-:|
|BASELINE|–|–|–|–|
|AUTOFE TRADICIONAL|✔|–|–|–|
|AUTOFE + LLM|✔|✔|–|–|
|AUTOFE COM INTERACOES|✔|–|✔|–|
|AUTOFE + INTERACOES + SELECAO|✔|–|✔|✔|

Validação cruzada de 5 folds, com os mesmos folds para todos os cenários (comparação pareada).

## Datasets e modelos executados

|Dataset|Tipo|Modelo|LLM utilizada|
|-|-|-|-|
|Titanic|classificação|Decision Tree|Groq `openai/gpt-oss-120b`|
|Breast Cancer|classificação|Decision Tree|Gemini (cenário LLM incompleto: 1/5 folds)|
|Diabetes|regressão|Random Forest|Groq `openai/gpt-oss-120b`|

O código também suporta Gradient Boosting e Dummy (ver `MODEL\_NAMES` em `src/main.py`), mas os resultados em `results/` cobrem apenas os modelos acima.

## Resultados

Resumo (métrica principal, média dos 5 folds). Detalhes, desvios, features e sugestões da LLM em [`results/report/RESULTADOS.md`](results/report/RESULTADOS.md).

|Cenário|Titanic (accuracy)|Breast Cancer (accuracy)|Diabetes (R²)|
|-|-|-|-|
|BASELINE|0.7912|0.9104|0.4294|
|AUTOFE TRADICIONAL|0.7834|0.9086|0.4294|
|AUTOFE + LLM|0.7811|0.9121 ⚠ incompleto|0.4342|
|AUTOFE COM INTERACOES|0.7632|0.9578|0.4279|
|AUTOFE + INTERACOES + SELECAO|0.7890|**0.9613**|0.4162|

Leitura honesta dos números:

* **Breast Cancer:** as interações trazem ganho consistente (+4,7 a +5,1 p.p., melhor que o baseline nos 5 folds).
* **Titanic e Diabetes:** nenhum cenário supera o baseline de forma convincente; as diferenças ficam dentro da variação entre folds.
* **LLM:** em Titanic e Diabetes o cenário com LLM ficou praticamente igual ao baseline (−0,010 accuracy e +0,005 R², ambos dentro do desvio). Não há evidência de ganho. Em Breast Cancer o cenário está incompleto e não deve ser interpretado.

## Instalação

```bash
git clone https://github.com/LucasBarros28/AutoFe-LLM.git
cd autofe
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env  (depois preencha as chaves)
```

## Execução

```bash
python -m src.main      # roda os experimentos configurados em src/main.py
python -m src.report    # gera results/report/RESULTADOS.md e os gráficos
pytest test/            # testes unitários
```

Configurações (dataset, modelo, provedor de LLM) ficam no topo de `src/main.py`. Os cenários com LLM exigem uma chave de API válida no `.env`.

## Limitações

* Apenas três transformações implementadas (o detector reconhece outliers e colunas constantes, mas ainda não os trata).
* Um único modelo por dataset nos resultados publicados; 5 folds e uma única semente, sem teste estatístico.
* Provedores de LLM diferentes entre datasets, e uma execução incompleta por esgotamento de cota.
* A LLM só recebe estatísticas das colunas, sem semântica de domínio detalhada.
* Respostas da LLM não são determinísticas; o cache torna reexecuções reprodutíveis apenas localmente.

## Status

* \[x] Análise exploratória, detecção de oportunidades e geração de features.
* \[x] Detecção de interações e seleção de features.
* \[x] Avaliação com validação cruzada e auditoria por fold.
* \[x] Integração com LLM (Groq e Gemini) com validação, cache e circuit breaker.
* \[x] Experimentos com múltiplos datasets e relatório consolidado.
* \[ ] Reexecutar Breast Cancer com a LLM completa (5/5 folds).
* \[ ] Rodar todos os modelos em todos os datasets.
* \[ ] Novas transformações (ex.: razões, binning, tratamento de outliers).

