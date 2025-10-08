import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import StringIO
import traceback
import streamlit as st
from scipy import stats
from typing import Dict, List, Any

# LangChain imports
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage

# Configurações
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ============================================

st.set_page_config(
    page_title="Agente EDA Autônomo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CLASSE DE GERENCIAMENTO DE DADOS
# ============================================

class DataManager:
    """Gerencia carregamento e acesso aos dados CSV"""

    def __init__(self):
        self.df = None
        self.filename = None
        self.load_history = []

    def load_csv(self, filepath):
        """Carrega arquivo CSV com tratamento de erros robusto"""
        try:
            # Tenta diferentes encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    self.df = pd.read_csv(filepath, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if self.df is None:
                return "Erro: Não foi possível decodificar o arquivo CSV"

            self.filename = os.path.basename(filepath)
            self.load_history.append({
                'timestamp': datetime.now(),
                'filename': self.filename,
                'shape': self.df.shape
            })

            return f"""✅ Arquivo carregado com sucesso!

📊 Informações básicas:
- Nome: {self.filename}
- Linhas: {self.df.shape[0]:,}
- Colunas: {self.df.shape[1]}
- Memória: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

📋 Colunas disponíveis:
{', '.join(self.df.columns.tolist())}

🔍 Tipos de dados:
{self.df.dtypes.value_counts().to_dict()}
"""
        except Exception as e:
            return f"❌ Erro ao carregar arquivo: {str(e)}"

    def get_dataframe(self):
        """Retorna o DataFrame atual"""
        return self.df

    def get_info(self):
        """Retorna informações sobre o dataset carregado"""
        if self.df is None:
            return "Nenhum arquivo carregado ainda."

        buffer = StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

# ============================================
# AMBIENTE PYTHON SEGURO
# ============================================

class SafePythonREPL:
    """Ambiente Python REPL seguro com acesso ao DataFrame"""

    def __init__(self):
        self.globals_dict = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'df': None,
        }
        self.execution_history = []
        self.last_plot_path = None

    def run(self, code: str) -> str:
        """Executa código Python com segurança"""
        try:
            self.last_plot_path = None
            
            # Atualiza referência ao DataFrame atual
            if 'data_manager' in st.session_state:
                self.globals_dict['df'] = st.session_state.data_manager.get_dataframe()

            if self.globals_dict['df'] is None:
                return "⚠️ Nenhum DataFrame carregado. Carregue um arquivo CSV primeiro."

            # Captura stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # Executa código
            exec(code, self.globals_dict)

            # Restaura stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # Registra histórico
            self.execution_history.append({
                'timestamp': datetime.now(),
                'code': code,
                'success': True
            })

            # Se gerou um plot, salva e retorna caminho
            if plt.get_fignums():
                plot_path = f"plot_{len(self.execution_history)}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                self.last_plot_path = plot_path
                
                # Adiciona o plot à lista de plots do Streamlit
                if 'plots' not in st.session_state:
                    st.session_state.plots = []
                st.session_state.plots.append(plot_path)
                
                plt.close('all')
                output += f"\n📊 Gráfico gerado!"

            return output if output else "✅ Código executado com sucesso (sem output)."

        except Exception as e:
            self.execution_history.append({
                'timestamp': datetime.now(),
                'code': code,
                'success': False,
                'error': str(e)
            })
            return f"❌ Erro na execução:\n{traceback.format_exc()}"
    
    def get_last_plot(self):
        """Retorna o caminho do último gráfico gerado"""
        return self.last_plot_path

# ============================================
# FERRAMENTAS PERSONALIZADAS
# ============================================

class DataAnalyzerTool:
    """Ferramenta avançada para análise exploratória de dados"""
    
    def __init__(self):
        self.last_analysis = None
    
    def get_basic_stats(self, df: pd.DataFrame) -> str:
        """Retorna estatísticas básicas do dataset"""
        try:
            stats_info = {
                'shape': df.shape,
                'column_types': {
                    'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
                    'categorical': df.select_dtypes(include=['object']).columns.tolist(),
                    'datetime': df.select_dtypes(include=['datetime']).columns.tolist()
                },
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'duplicated_rows': df.duplicated().sum()
            }
            
            # Estatísticas para colunas numéricas
            numeric_stats = {}
            for col in stats_info['column_types']['numeric']:
                numeric_stats[col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75)),
                    'skewness': float(stats.skew(df[col].dropna())),
                    'kurtosis': float(stats.kurtosis(df[col].dropna()))
                }
            
            stats_info['numeric_stats'] = numeric_stats
            
            # Formata output
            output = f"""
📊 ESTATÍSTICAS BÁSICAS DO DATASET

📏 Dimensões: {stats_info['shape'][0]:,} linhas x {stats_info['shape'][1]} colunas
💾 Uso de memória: {stats_info['memory_usage'] / 1024**2:.2f} MB
🔄 Linhas duplicadas: {stats_info['duplicated_rows']}

📋 TIPOS DE COLUNAS:
  • Numéricas: {len(stats_info['column_types']['numeric'])} - {', '.join(stats_info['column_types']['numeric'])}
  • Categóricas: {len(stats_info['column_types']['categorical'])} - {', '.join(stats_info['column_types']['categorical'])}
  • Data/Hora: {len(stats_info['column_types']['datetime'])} - {', '.join(stats_info['column_types']['datetime'])}

❌ VALORES AUSENTES:
"""
            missing = {k: v for k, v in stats_info['missing_values'].items() if v > 0}
            if missing:
                for col, count in missing.items():
                    pct = (count / stats_info['shape'][0]) * 100
                    output += f"  • {col}: {count} ({pct:.2f}%)\n"
            else:
                output += "  ✅ Nenhum valor ausente!\n"
            
            output += "\n📈 ESTATÍSTICAS NUMÉRICAS:\n"
            for col, stats_dict in numeric_stats.items():
                output += f"\n  {col}:\n"
                output += f"    Média: {stats_dict['mean']:.2f} | Mediana: {stats_dict['median']:.2f}\n"
                output += f"    Desvio Padrão: {stats_dict['std']:.2f}\n"
                output += f"    Mín: {stats_dict['min']:.2f} | Máx: {stats_dict['max']:.2f}\n"
                output += f"    Q1: {stats_dict['q25']:.2f} | Q3: {stats_dict['q75']:.2f}\n"
                output += f"    Assimetria: {stats_dict['skewness']:.3f} | Curtose: {stats_dict['kurtosis']:.3f}\n"
            
            return output
            
        except Exception as e:
            return f"❌ Erro ao calcular estatísticas: {str(e)}"
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> str:
        """Detecta outliers usando diferentes métodos"""
        try:
            outliers = {}
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return "⚠️ Nenhuma coluna numérica encontrada para detecção de outliers."
            
            for col in numeric_columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outliers[col] = {
                        'count': outlier_mask.sum(),
                        'percentage': (outlier_mask.sum() / len(df)) * 100,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'indices': df[outlier_mask].index.tolist()[:10]  # Primeiros 10
                    }
                
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    threshold = 3
                    outlier_mask = z_scores > threshold
                    outliers[col] = {
                        'count': outlier_mask.sum(),
                        'percentage': (outlier_mask.sum() / len(df)) * 100,
                        'threshold': threshold,
                        'indices': df[outlier_mask].index.tolist()[:10]
                    }
            
            # Formata output
            output = f"🔍 DETECÇÃO DE OUTLIERS (Método: {method.upper()})\n\n"
            
            for col, info in outliers.items():
                if info['count'] > 0:
                    output += f"📊 {col}:\n"
                    output += f"  • Outliers encontrados: {info['count']} ({info['percentage']:.2f}%)\n"
                    if method == 'iqr':
                        output += f"  • Limites: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]\n"
                    else:
                        output += f"  • Threshold Z-score: {info['threshold']}\n"
                    if info['indices']:
                        output += f"  • Primeiros índices: {info['indices']}\n"
                    output += "\n"
            
            if not any(info['count'] > 0 for info in outliers.values()):
                output += "✅ Nenhum outlier detectado!\n"
            
            return output
            
        except Exception as e:
            return f"❌ Erro ao detectar outliers: {str(e)}"
    
    def calculate_correlations(self, df: pd.DataFrame, threshold: float = 0.5) -> str:
        """Calcula e analisa correlações entre variáveis numéricas"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return "⚠️ É necessário pelo menos 2 colunas numéricas para calcular correlações."
            
            corr_matrix = numeric_df.corr()
            
            # Encontra correlações fortes
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            
            # Ordena por valor absoluto
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Formata output
            output = f"🔗 ANÁLISE DE CORRELAÇÕES (threshold: {threshold})\n\n"
            output += f"📊 Matriz de correlação: {len(numeric_df.columns)}x{len(numeric_df.columns)}\n"
            output += f"🔍 Correlações fortes encontradas: {len(strong_correlations)}\n\n"
            
            if strong_correlations:
                output += "📈 CORRELAÇÕES MAIS FORTES:\n"
                for i, corr in enumerate(strong_correlations[:10], 1):  # Top 10
                    strength = "Muito forte" if abs(corr['correlation']) > 0.8 else "Forte"
                    direction = "positiva" if corr['correlation'] > 0 else "negativa"
                    output += f"{i}. {corr['var1']} ↔ {corr['var2']}\n"
                    output += f"   Correlação: {corr['correlation']:.3f} ({strength} {direction})\n\n"
            else:
                output += "ℹ️ Nenhuma correlação forte encontrada acima do threshold.\n"
            
            # Adiciona matriz completa
            output += "\n📋 MATRIZ DE CORRELAÇÃO COMPLETA:\n"
            output += corr_matrix.to_string()
            
            return output
            
        except Exception as e:
            return f"❌ Erro ao calcular correlações: {str(e)}"
    
    def analyze_categorical(self, df: pd.DataFrame, top_n: int = 10) -> str:
        """Analisa variáveis categóricas"""
        try:
            cat_columns = df.select_dtypes(include=['object']).columns
            
            if len(cat_columns) == 0:
                return "⚠️ Nenhuma coluna categórica encontrada."
            
            output = f"📊 ANÁLISE DE VARIÁVEIS CATEGÓRICAS\n"
            output += f"Total de colunas: {len(cat_columns)}\n\n"
            
            for col in cat_columns:
                unique_count = df[col].nunique()
                most_common = df[col].value_counts().head(top_n)
                missing = df[col].isnull().sum()
                
                output += f"📋 {col}:\n"
                output += f"  • Valores únicos: {unique_count}\n"
                output += f"  • Valores ausentes: {missing} ({(missing/len(df)*100):.2f}%)\n"
                output += f"  • Valores mais frequentes (top {min(top_n, len(most_common))}):\n"
                
                for value, count in most_common.items():
                    pct = (count / len(df)) * 100
                    output += f"    - {value}: {count} ({pct:.2f}%)\n"
                
                output += "\n"
            
            return output
            
        except Exception as e:
            return f"❌ Erro ao analisar variáveis categóricas: {str(e)}"
    
    def run_analysis(self, analysis_type: str) -> str:
        """Executa análise baseada no tipo solicitado"""
        if 'data_manager' not in st.session_state:
            return "⚠️ DataManager não inicializado."
        
        df = st.session_state.data_manager.get_dataframe()
        
        if df is None:
            return "⚠️ Nenhum DataFrame carregado."
        
        analysis_type = analysis_type.lower().strip()
        
        if 'basic' in analysis_type or 'estatistic' in analysis_type or 'resumo' in analysis_type:
            return self.get_basic_stats(df)
        
        elif 'outlier' in analysis_type:
            method = 'zscore' if 'zscore' in analysis_type or 'z-score' in analysis_type else 'iqr'
            return self.detect_outliers(df, method)
        
        elif 'correla' in analysis_type:
            threshold = 0.3 if 'fraca' in analysis_type else 0.5
            return self.calculate_correlations(df, threshold)
        
        elif 'categor' in analysis_type:
            return self.analyze_categorical(df)
        
        else:
            return f"""ℹ️ Tipo de análise não reconhecido: '{analysis_type}'

Análises disponíveis:
- basic/estatisticas/resumo: Estatísticas básicas completas
- outlier/outliers: Detecção de outliers (IQR ou Z-score)
- correlacao/correlações: Análise de correlações
- categorica/categoricas: Análise de variáveis categóricas
"""

# Instância global da ferramenta de análise
data_analyzer = DataAnalyzerTool()

def load_csv_tool(filepath: str) -> str:
    """Ferramenta para carregar arquivo CSV"""
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    result = st.session_state.data_manager.load_csv(filepath)
    return result

def get_dataframe_info_tool(dummy: str = "") -> str:
    """Retorna informações sobre o DataFrame carregado"""
    if 'data_manager' not in st.session_state or st.session_state.data_manager.df is None:
        return "Nenhum arquivo CSV foi carregado ainda. Use a ferramenta 'load_csv' primeiro."
    return st.session_state.data_manager.get_info()

# ============================================
# SISTEMA DE PROMPT DO AGENTE
# ============================================

SYSTEM_PROMPT = """Você é um Cientista de Dados Especialista em Análise Exploratória de Dados (EDA).

🎯 SUA MISSÃO:
Ajudar usuários a explorar, analisar e extrair insights de arquivos CSV através de análises estatísticas e visualizações.

📊 DATASET ATUAL:
O DataFrame está disponível na variável 'df'. Use-o diretamente em seus códigos Python.

🛠️ FERRAMENTAS DISPONÍVEIS:

1. **get_dataframe_info**: Informações básicas sobre o dataset
   - Use PRIMEIRO quando carregar dados novos
   - Mostra colunas, tipos, shape

2. **data_analyzer**: Análises estatísticas avançadas (USE ESTA PRIMEIRO para análises)
   - "basic" → Estatísticas completas (média, mediana, std, assimetria, curtose, missing values)
   - "outlier" → Detecção de outliers (IQR ou Z-score)
   - "correlacao" → Correlações entre variáveis numéricas
   - "categorica" → Análise de variáveis categóricas
   
3. **python_repl**: Código Python customizado
   - Use quando data_analyzer não cobrir a análise necessária
   - Para visualizações
   - Para análises temporais
   - Para transformações específicas

⚡ FLUXO DE TRABALHO OTIMIZADO:

**Para perguntas sobre estatísticas/resumo:**
1. Use `data_analyzer` com "basic"
2. Interprete os resultados
✅ Rápido e completo!

**Para perguntas sobre outliers:**
1. Use `data_analyzer` com "outlier"
2. Se precisar visualizar, use `python_repl` para gráfico
✅ Análise + visualização!

**Para perguntas sobre correlação:**
1. Use `data_analyzer` com "correlacao"
2. Se precisar heatmap, use `python_repl`
✅ Números + gráfico!

**Para análise de variáveis categóricas:**
1. Use `data_analyzer` com "categorica"
2. Se precisar gráficos de barras, use `python_repl`

**Para visualizações ou análises customizadas:**
1. Use `python_repl` diretamente

📋 EXEMPLOS DE USO:

Pergunta: "Quais as estatísticas básicas?"
→ Use: data_analyzer("basic")
→ Resultado: Estatísticas completas em segundos!

Pergunta: "Existem outliers?"
→ Use: data_analyzer("outlier")
→ Depois (opcional): python_repl para visualizar

Pergunta: "Quais variáveis estão correlacionadas?"
→ Use: data_analyzer("correlacao")
→ Depois (opcional): python_repl para heatmap

Pergunta: "Valores mais frequentes?"
→ Use: data_analyzer("categorica")

Pergunta: "Tendências temporais?"
→ Use: python_repl (análise customizada)

Pergunta: "Faça gráfico de X"
→ Use: python_repl

⚠️ REGRAS CRÍTICAS:
1. **SEMPRE prefira data_analyzer** para análises estatísticas padrão
2. **Use python_repl** apenas para visualizações ou análises não-padrão
3. **SEJA DIRETO**: Execute ferramentas imediatamente, não explique o que VAI fazer
4. **MOSTRE RESULTADOS**: Sempre apresente os números e interprete
5. **NÃO invente dados**: Use apenas o que as ferramentas retornam

💡 DICAS DE EFICIÊNCIA:
- data_analyzer é mais rápido que python_repl para análises padrão
- Combine ferramentas: data_analyzer para números + python_repl para gráficos
- Uma ferramenta por vez: execute, analise resultado, depois decida próximo passo

🧠 LEMBRE-SE:
Você é um agente de AÇÃO, não apenas de planejamento. Execute, mostre resultados, interprete.
"""

# ============================================
# INICIALIZAÇÃO DO AGENTE
# ============================================

@st.cache_resource
def create_agent(api_key, max_iter=15):
    """Cria e configura o agente LangChain"""
    
    # Configura a API key
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Modelo OpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=4000
    )

    # Memória conversacional
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )
    
    # Instância do Python REPL
    if 'python_repl' not in st.session_state:
        st.session_state.python_repl = SafePythonREPL()

    # Ferramentas disponíveis
    tools = [
        Tool(
            name="get_dataframe_info",
            func=get_dataframe_info_tool,
            description="""Retorna informações detalhadas sobre o DataFrame carregado.
            Use PRIMEIRO para conhecer as colunas, tipos de dados e estatísticas básicas.
            Input: qualquer string (será ignorada)
            Output: informações completas do dataset"""
        ),
        Tool(
            name="data_analyzer",
            func=data_analyzer.run_analysis,
            description="""Ferramenta avançada para análise exploratória de dados.
            
            Tipos de análise disponíveis:
            - "basic" ou "estatisticas" ou "resumo": Estatísticas descritivas completas
              (média, mediana, desvio padrão, assimetria, curtose, valores ausentes)
            
            - "outlier" ou "outliers": Detecção de outliers
              - Use "outlier iqr" para método IQR (padrão)
              - Use "outlier zscore" para método Z-score
            
            - "correlacao" ou "correlações": Análise de correlações entre variáveis numéricas
              - Use "correlacao fraca" para threshold 0.3
              - Padrão: threshold 0.5
            
            - "categorica" ou "categoricas": Análise de variáveis categóricas
              (valores únicos, frequências, valores mais comuns)
            
            Input: tipo de análise desejada (string)
            Output: análise detalhada formatada
            
            Exemplos de uso:
            - "basic" → Estatísticas completas
            - "outlier iqr" → Outliers pelo método IQR
            - "correlacao" → Correlações fortes (>0.5)
            - "categorica" → Análise das variáveis categóricas"""
        ),
        Tool(
            name="python_repl",
            description="""Executa código Python para análise de dados e visualizações.
            O DataFrame está disponível como 'df' (já carregado e pronto para uso).
            
            Bibliotecas disponíveis: 
            - pandas (pd)
            - numpy (np) 
            - matplotlib.pyplot (plt)
            - seaborn (sns)
            
            Use para:
            - Análises customizadas não cobertas pelo data_analyzer
            - Visualizações: histogramas, boxplots, scatter plots, heatmaps
            - Transformações de dados
            - Análises temporais
            - Filtros e agregações específicas
            
            IMPORTANTE: Sempre use print() para mostrar resultados!
            
            Input: código Python válido
            Output: resultado da execução + gráficos gerados"""
        )
    ]

    # Inicializa agente
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=max_iter,  # Configurável
        max_execution_time=120,
        early_stopping_method="generate",
        agent_kwargs={
            "system_message": SystemMessage(content=SYSTEM_PROMPT)
        }
    )

    return agent

# ============================================
# INICIALIZAÇÃO DO SESSION STATE
# ============================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

if 'plots' not in st.session_state:
    st.session_state.plots = []

if 'python_repl' not in st.session_state:
    st.session_state.python_repl = SafePythonREPL()

# ============================================
# INTERFACE STREAMLIT
# ============================================

# Header
st.title("🤖 Agente Autônomo para Análise Exploratória de Dados")
st.markdown("### 📊 Powered by LangChain + OpenAI GPT-4")

st.markdown("""
**Como usar:**
1. Configure sua API Key da OpenAI na barra lateral
2. Faça upload do seu arquivo CSV
3. Faça perguntas sobre os dados em linguagem natural
4. Receba análises, estatísticas e visualizações
""")

st.divider()

api_key = st.secrets["OPENAI_API_KEY"]

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # API Key
    # api_key = st.text_input(
    #     "OpenAI API Key",
    #     type="password",
    #     help="Cole sua chave API da OpenAI aqui"
    # )
    
    if api_key:
        st.success("✅ API Key configurada!")
        
        # Configuração de iterações máximas
        max_iterations = st.slider(
            "Máximo de iterações do agente",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="Número máximo de passos que o agente pode executar. Aumente para análises mais complexas."
        )
        
        # Cria o agente com a API key e max_iterations
        if 'agent' not in st.session_state or st.session_state.get('api_key') != api_key or st.session_state.get('max_iter') != max_iterations:
            with st.spinner("Inicializando agente..."):
                st.session_state.agent = create_agent(api_key, max_iterations)
                st.session_state.api_key = api_key
                st.session_state.max_iter = max_iterations
    else:
        st.warning("⚠️ Configure sua API Key para começar")
    
    st.divider()
    
    # Informações sobre ferramentas
    with st.expander("🛠️ Ferramentas do Agente"):
        st.markdown("""
        **Data Analyzer (Análises Rápidas):**
        - ⚡ Estatísticas básicas completas
        - 🔗 Análise de correlações
        - 📊 Análise de variáveis categóricas
        
        **Python REPL (Análises Customizadas):**
        - 📈 Visualizações e gráficos
        - 🔍 Transformações específicas
        - 🎨 Análises não-padrão
        
        💡 O agente escolhe automaticamente a melhor ferramenta!
        """)
    
    # Dicas de uso
    with st.expander("💡 Dicas para usar o agente"):
        st.markdown("""
        **Perguntas rápidas (1-3 iterações):**
        - "Mostre estatísticas básicas"
        - "Analise correlações"
        - "Variáveis categóricas"
        
        **Análises médias (5-10 iterações):**
        - "Analise correlação e crie heatmap"
        - "Detecte outliers e visualize"
        - "Compare grupos A e B"
        
        **Análises complexas (10-20 iterações):**
        - "Análise exploratória completa com gráficos"
        - "Identifique padrões temporais e tendências"
        - "Análise completa de todas as variáveis"
        
        ⚡ **Nova ferramenta Data Analyzer = Respostas mais rápidas!**
        """)
    
    st.divider()
    
    # Upload de arquivo
    st.header("📁 Upload de Dados")
    uploaded_file = st.file_uploader(
        "Selecione um arquivo CSV",
        type=['csv'],
        help="Faça upload do arquivo CSV que deseja analisar"
    )
    
    if uploaded_file is not None:
        # Salva o arquivo temporariamente
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Carregar CSV", use_container_width=True):
            with st.spinner("Carregando arquivo..."):
                result = st.session_state.data_manager.load_csv(temp_path)
                st.success("Arquivo carregado!")
                with st.expander("📊 Informações do Dataset"):
                    st.text(result)
    
    st.divider()
    
    # Botão para limpar chat
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.plots = []
        st.rerun()
    
    # Informações do dataset
    if st.session_state.data_manager.df is not None:
        st.divider()
        st.header("📊 Dataset Info")
        df = st.session_state.data_manager.df
        st.metric("Linhas", f"{df.shape[0]:,}")
        st.metric("Colunas", df.shape[1])
        st.metric("Memória", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversa com o Agente")
    
    # Perguntas sugeridas
    if st.session_state.data_manager.df is not None and len(st.session_state.messages) == 0:
        st.info("💡 **Perguntas sugeridas para começar:**")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📊 Estatísticas básicas", use_container_width=True):
                st.session_state.suggested_question = "Mostre as estatísticas básicas do dataset"
        
        with col_b:
            if st.button("🔍 Detectar outliers", use_container_width=True):
                st.session_state.suggested_question = "Detecte outliers nas variáveis numéricas"
        
        with col_c:
            if st.button("🔗 Análise de correlação", use_container_width=True):
                st.session_state.suggested_question = "Analise as correlações entre variáveis e mostre as mais fortes"
        
        col_d, col_e, col_f = st.columns(3)
        
        with col_d:
            if st.button("📋 Variáveis categóricas", use_container_width=True):
                st.session_state.suggested_question = "Analise as variáveis categóricas e mostre os valores mais frequentes"
        
        with col_e:
            if st.button("📈 Análise exploratória completa", use_container_width=True):
                st.session_state.suggested_question = "Faça uma análise exploratória completa incluindo estatísticas, outliers e correlações"
        
        with col_f:
            if st.button("📉 Visualizações", use_container_width=True):
                st.session_state.suggested_question = "Crie visualizações das principais variáveis do dataset"
    
    # Container para mensagens
    chat_container = st.container(height=500)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input de mensagem
    prompt = None
    
    # Verifica se há pergunta sugerida
    if 'suggested_question' in st.session_state:
        prompt = st.session_state.suggested_question
        del st.session_state.suggested_question
    else:
        prompt = st.chat_input("Digite sua pergunta sobre os dados...", disabled=not api_key)
    
    if prompt:
        if 'agent' not in st.session_state:
            st.error("❌ Configure sua API Key primeiro!")
        elif st.session_state.data_manager.df is None:
            st.warning("⚠️ Carregue um arquivo CSV primeiro antes de fazer perguntas!")
        else:
            # Adiciona mensagem do usuário
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Processa resposta do agente
                with st.chat_message("assistant"):
                    with st.spinner("Analisando..."):
                        try:
                            response = st.session_state.agent.run(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_str = str(e)
                            
                            # Verifica se é erro de limite de iterações
                            if "iteration limit" in error_str.lower() or "time limit" in error_str.lower():
                                error_msg = """⚠️ **Análise muito complexa!**
                                
O agente atingiu o limite de iterações. Isso pode acontecer quando:
- A análise requer muitos passos
- Há muitas variáveis para processar
- A pergunta é muito abrangente

**Sugestões:**
1. Aumente o limite de iterações na sidebar (atualmente: {})
2. Divida sua pergunta em partes menores
3. Seja mais específico sobre o que deseja analisar

💡 Tentarei responder com o que consegui processar até agora...
                                """.format(st.session_state.get('max_iter', 15))
                                
                                st.warning(error_msg)
                                
                                # Tenta obter resposta parcial se houver
                                if hasattr(st.session_state.agent, 'agent') and hasattr(st.session_state.agent.agent, 'return_values'):
                                    partial = st.session_state.agent.agent.return_values.get('output', 'Análise incompleta.')
                                    st.markdown(partial)
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg + "\n\n" + partial})
                                else:
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            else:
                                # Outros erros
                                error_msg = f"❌ **Erro ao processar:**\n```\n{error_str}\n```"
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()

with col2:
    st.header("📊 Gráficos Gerados")
    
    if st.session_state.plots:
        for i, plot_path in enumerate(reversed(st.session_state.plots)):
            if os.path.exists(plot_path):
                st.image(plot_path, caption=f"Gráfico {len(st.session_state.plots) - i}", use_container_width=True)
                st.divider()
    else:
        st.info("Nenhum gráfico gerado ainda. Peça ao agente para criar visualizações!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    Desenvolvido com ❤️ usando LangChain e Streamlit
</div>
""", unsafe_allow_html=True)