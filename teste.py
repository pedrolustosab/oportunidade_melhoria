import os
from process_analyser import analyze_single_process
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

processo = [{
        "ramo_empresa": "Moda",
        "direcionadores": "Aumentar lucro",
        "nome_processo": "Otimização",
        "atividade": "Otimaização de processo",
        "evento": "Otimaização",
        "causa": "Aumento do preço dos insumos",
        "operaciona_atividade": "",
        "sistema_relacionado": "",
        "solucao_gap": "",
        "outro_gap": ""
}]

resultados = analyze_single_process(
    process_data=processo,
    faiss_index_path="process_index.faiss",
    api_key=api_key
)

print(resultados)