# 🤖 Agente Autônomo para Análise Exploratória de Dados

Sistema inteligente de análise de dados usando LangChain, OpenAI GPT-4 e Streamlit. O agente executa análises estatísticas avançadas, detecta padrões, identifica outliers e gera visualizações automaticamente.

## ✨ Funcionalidades

### 🔬 Análises Estatísticas Avançadas
- **Estatísticas Descritivas Completas**: Média, mediana, desvio padrão, assimetria, curtose
- **Detecção de Outliers**: Métodos IQR e Z-score
- **Análise de Correlações**: Matriz de correlação e identificação de relações fortes
- **Análise de Variáveis Categóricas**: Frequências, valores únicos, distribuições

### 📊 Visualizações Automáticas
- Histogramas e boxplots
- Heatmaps de correlação
- Gráficos de dispersão
- Análises temporais
- Gráficos customizados

### 🤖 Agente Inteligente
- Interpreta perguntas em linguagem natural
- Escolhe automaticamente a melhor ferramenta
- Executa análises e gera insights
- Mantém contexto da conversa

## 🚀 Instalação

### 1. Criar ambiente virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar API Key da OpenAI

Você precisará de uma chave API da OpenAI. Obtenha em: https://platform.openai.com/api-keys

## 📖 Como Usar

### 1. Iniciar a aplicação

```bash
streamlit run eda_agent_app.py
```

### 2. Configurar API Key

Na barra lateral, cole sua chave API da OpenAI.

### 3. Fazer upload do CSV

Clique em "Selecione um arquivo CSV" e escolha seu dataset.

### 4. Fazer perguntas

Use as perguntas sugeridas ou digite suas próprias perguntas em linguagem natural:

**Exemplos:**
- "Mostre as estatísticas básicas do dataset"
- "Existem outliers nas variáveis numéricas?"
- "Quais variáveis estão mais correlacionadas?"
- "Analise as variáveis categóricas"
- "Crie um heatmap de correlações"
- "Faça uma análise exploratória completa"

## 🛠️ Ferramentas do Agente

### Data Analyzer Tool
Ferramenta otimizada para análises estatísticas rápidas:
- ⚡ Estatísticas básicas
- 🔗 Análise de correlações
- 📊 Análise de variáveis categóricas

### Python REPL Tool
Para análises customizadas e visualizações:
- 📈 Gráficos e visualizações
- 🔍 Transformações de dados
- 🎨 Análises específicas

## ⚙️ Configurações Avançadas

### Ajustar Limite de Iterações

Use o slider na barra lateral para controlar quantos passos o agente pode executar:
- **5-10 iterações**: Análises simples e rápidas
- **10-20 iterações**: Análises médias com visualizações
- **20-30 iterações**: Análises complexas e completas

### Timeouts

O agente tem um timeout de 120 segundos (2 minutos) por análise para evitar travamentos.

## 📊 Tipos de Análises Suportadas

### Estatísticas Descritivas
- Medidas de tendência central
- Medidas de dispersão
- Assimetria e curtose
- Valores ausentes
- Distribuições

### Detecção de Anomalias
- Outliers por IQR (Interquartile Range)
- Outliers por Z-score
- Análise de impacto
- Visualização de outliers

### Análise de Relações
- Correlação de Pearson
- Correlação de Spearman
- Identificação de multicolinearidade
- Heatmaps de correlação

### Análise de Variáveis Categóricas
- Frequências absolutas e relativas
- Valores únicos
- Valores mais/menos comuns
- Gráficos de barras

### Análises Temporais
- Tendências
- Sazonalidade
- Padrões ao longo do tempo
- Gráficos de linha

## 🔧 Troubleshooting

### Erro: "Agent stopped due to iteration limit"
**Solução:** Aumente o limite de iterações no slider da sidebar ou divida a pergunta em partes menores.

### Erro: "API Key inválida"
**Solução:** Verifique se copiou a chave corretamente e se ela está ativa em sua conta OpenAI.

### Erro ao carregar CSV
**Solução:** Verifique o encoding do arquivo. O sistema tenta múltiplos encodings automaticamente (UTF-8, Latin-1, ISO-8859-1, CP1252).

### Gráficos não aparecem
**Solução:** Os gráficos aparecem na coluna direita. Role a página para baixo se necessário.

## 📦 Dependências Principais

- **streamlit**: Interface web interativa
- **langchain**: Framework para agentes de IA
- **langchain-openai**: Integração com GPT-4
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **scipy**: Análises estatísticas
- **matplotlib/seaborn**: Visualizações

## 🤝 Contribuindo

Sugestões de melhorias são bem-vindas! Algumas ideias:
- Suporte para mais formatos (Excel, JSON, Parquet)
- Análises de machine learning
- Exportação de relatórios em PDF
- Testes estatísticos automatizados
- Integração com bancos de dados

## 📝 Licença

Este projeto é de código aberto. Use livremente para fins educacionais e comerciais.

## 🆘 Suporte

Para dúvidas ou problemas, verifique:
1. A documentação da OpenAI: https://platform.openai.com/docs
2. A documentação do LangChain: https://python.langchain.com/docs
3. A documentação do Streamlit: https://docs.streamlit.io

## 🌟 Exemplo de Uso Completo

```python
# 1. Inicie a aplicação
streamlit run eda_agent_app.py

# 2. Configure API Key na sidebar

# 3. Faça upload do CSV

# 4. Perguntas sugeridas:
"Mostre estatísticas básicas"
→ Obtém estatísticas completas em segundos

"Detecte outliers"
→ Identifica valores anômalos usando IQR

"Analise correlações e crie heatmap"
→ Matriz de correlação + visualização

"Faça análise exploratória completa"
→ Estatísticas + outliers + correlações + visualizações
```

---

**Desenvolvido com ❤️ usando LangChain, OpenAI GPT-4 e Streamlit**