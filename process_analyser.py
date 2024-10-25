import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import logging

class ProcessAnalyzer:
    def __init__(self, faiss_index_path: str = "process_index.faiss", api_key: str = None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._setup_api_key(api_key)
        self.vectorstore = self._load_faiss_index(faiss_index_path)
        self.chat = self._initialize_llm()
        self.retrieval_chain = self._setup_retrieval_chain()
        self.chat_history = []  


    def _setup_api_key(self, api_key: str = None) -> None:
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key not found. Please provide it or set it in the .env file.")

    def _load_faiss_index(self, faiss_index_path: str) -> FAISS:
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at path: {faiss_index_path}")
        return FAISS.load_local(
            faiss_index_path, 
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )

    def _initialize_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=2000)

    def _setup_retrieval_chain(self) -> ConversationalRetrievalChain:
        return ConversationalRetrievalChain.from_llm(
            llm=self.chat,
            retriever=self.vectorstore.as_retriever(),
            memory=None,
            verbose=True
        )

    def analyze_process(self, process_data:list[dict]) -> pd.DataFrame:
        """
        Analisa um processo de negócio e sugere melhorias.
        
        Args:
            process_data (dict): Dicionário contendo informações do processo.
            
        Returns:
            pd.DataFrame: DataFrame com análise e sugestões de melhoria.
        """
        # Convert input data to DataFrame
        df_input = pd.DataFrame(process_data)
        
        # Create combined text for analysis
        input_columns = df_input.columns.tolist()
        df_input['combined_text'] = df_input.apply(
            lambda row: ' '.join(f"{col}: {row[col]}" for col in input_columns), 
            axis=1
        )
        process_text = df_input['combined_text'].to_string(index=False)
        
        analysis_prompt = f"""
        Você é um consultor sênior especializado em otimização de processos de uma das principais empresas globais de consultoria estratégica. Com base em sua vasta experiência em projetos de transformação organizacional e melhoria contínua, analise criteriosamente o processo a seguir:

        PROCESSO ANALISADO:
        {process_text}

        DIRETRIZES DE ANÁLISE:
        - Utilize metodologias como Lean Six Sigma, Theory of Constraints e Business Process Management
        - Considere as melhores práticas de mercado no setor específico
        - Avalie aspectos de transformação digital e Industry 4.0
        - Considere impactos em custos, qualidade, tempo e satisfação do cliente
        - Analise riscos e pontos de controle necessários

        ASPECTOS ESPECÍFICOS A SEREM CONSIDERADOS:
        1. Gargalos e redundâncias
        2. Oportunidades de automação (RPA, IA, etc.)
        3. Integração entre sistemas e áreas
        4. Compliance e gestão de riscos
        5. Indicadores de performance (KPIs)
        6. Capacitação necessária
        7. Impacto nas pessoas e mudança cultural

        ESTRUTURA DA RESPOSTA REQUISITADA:
        Retorne uma lista de dicionários em Python contendo exatamente os seguintes campos:

        [
            {{
                "oportunidade_melhoria": "Descrição clara e específica da oportunidade identificada",
                "tarefa": "Ação concreta e mensurável para implementar a melhoria",
                "criterio_aceitacao": "Métricas e resultados específicos que indicam o sucesso da implementação"
            }},
            # ... outras oportunidades de melhoria
        ]

        FORMATO DE SAÍDA:
        Retorne exclusivamente a lista de dicionários Python, sem comentários adicionais ou explicações.
        Se estiver em string transforme em lista de dicionários Python
        Sem aspas
        Sem "python"
        
        """
        # Get single comprehensive response
        result = self.retrieval_chain(
            {"question": analysis_prompt, "chat_history": self.chat_history}
        )["answer"]
        json_result = json.loads(result)
        df_result = pd.DataFrame(json_result)
        return df_result
    
def analyze_single_process(process_data: dict, faiss_index_path: str = "process_index.faiss", api_key: str = None) -> pd.DataFrame:
    """
    Função auxiliar para analisar um único processo de forma simplificada.
    """
    analyzer = ProcessAnalyzer(faiss_index_path, api_key)
    return analyzer.analyze_process(process_data)
