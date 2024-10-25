import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging

class ProcessEmbeddingsCreator:
    """
    Classe responsável por criar e salvar embeddings de processos usando FAISS.
    """
    
    def __init__(self, excel_path: str, faiss_index_path: str = "process_index.faiss", api_key: str = None):
        """
        Inicializa o ProcessEmbeddingsCreator.

        Args:
            excel_path (str): Caminho para o arquivo Excel.
            faiss_index_path (str): Caminho para salvar o índice FAISS.
            api_key (str, optional): Chave da API OpenAI.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.excel_path = excel_path
        self.faiss_index_path = faiss_index_path
        
        self._setup_api_key(api_key)
        self.df = self._load_excel_data()
        self.vectorstore = self._create_vectorstore()

    def _setup_api_key(self, api_key: str = None) -> None:
        """Configura a chave da API OpenAI."""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("Chave da API OpenAI não encontrada. Forneça-a ou configure no arquivo .env.")

    def _load_excel_data(self) -> pd.DataFrame:
        """Carrega e pré-processa os dados do Excel."""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='Vfinal', skiprows=3)
            df = df.loc[:, ~df.columns.str.contains('Unnamed:')]

            columns = [
                'SEGMENTO DE MERCADO', 'GANHOS/OBJETIVO', 'PROCESSO',
                'ATIVIDADE RELACIONADA', 'CAUSA', 'Pessoas / Organização',
                'DESCONEXÕES (GAP)', 'MELHORIA/SOLUÇÃO'
            ]

            new_names = [
                'ramo_empresa', 'direcionadores', 'nome_processo',
                'atividade', 'causa', 'operaciona_atividade',
                'solucao_gap', 'melhoria'
            ]

            df = df[columns]
            df.columns = new_names
            df['combined_text'] = df.apply(lambda row: ' '.join(f"{col}: {row[col]}" for col in new_names), axis=1)
            
            return df
        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivo Excel: {str(e)}")
            raise

    def _create_vectorstore(self) -> FAISS:
        """Cria e inicializa o FAISS vector store."""
        loader = DataFrameLoader(self.df, page_content_column='combined_text')
        documents = loader.load()
        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(documents, embeddings)

    def save_embeddings(self):
        """Salva o índice FAISS em um arquivo."""
        self.vectorstore.save_local(self.faiss_index_path)
        self.logger.info(f"Índice FAISS salvo em {self.faiss_index_path}")

def create_and_save_embeddings(excel_path: str, faiss_index_path: str = "process_index.faiss", api_key: str = None):
    """
    Função auxiliar para criar e salvar embeddings de forma simplificada.
    
    Args:
        excel_path (str): Caminho para o arquivo Excel.
        faiss_index_path (str): Caminho para salvar o índice FAISS.
        api_key (str, optional): Chave da API OpenAI.
    """
    creator = ProcessEmbeddingsCreator(excel_path, faiss_index_path, api_key)
    creator.save_embeddings()
    
create_and_save_embeddings(
    excel_path="Base.xlsx",
    faiss_index_path="process_index.faiss",
)