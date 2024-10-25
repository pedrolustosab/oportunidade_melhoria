import streamlit as st
import base64
from pathlib import Path
import os
from process_analyser import analyze_single_process
from dotenv import load_dotenv
import pandas as pd


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def convert_df_to_csv(df):
    """
    Convert a pandas DataFrame to a CSV string with custom separators and encoding.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        str: A CSV string representation of the DataFrame.
    """
    csv = df.to_csv(index=False, sep="|", encoding='utf-8')
    return csv

def stylable_container(key, css_styles):
    """
    Create a stylable container with custom CSS.

    Args:
    key (int): The index of the child div to style.
    css_styles (str): CSS styles to apply to the container.

    Returns:
    streamlit.container: A stylable Streamlit container.
    """
    st.markdown(f"""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child({key}) {{
            {css_styles}
        }}
        </style>
    """, unsafe_allow_html=True)
    return st.container()

def add_bg_from_local(image_file):
    """
    Add a background image to the Streamlit app from a local file.

    Args:
    image_file (str): Path to the local image file.
    """
    with Path(image_file).open("rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def load_css(css_file):
    """
    Load and apply CSS from a file to the Streamlit app.

    Args:
    css_file (str): Path to the CSS file.
    """
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_button_style(button_class):
    """
    Get the button style based on its class.

    Args:
    button_class (str): The class of the button ('current', 'previous', or other).

    Returns:
    str: CSS style for the button.
    """
    if button_class == "current":
        return "background-color: #01374C; color: white; font-weight: bold; font-size: 24px;"
    elif button_class == "previous":
        return "background-color: #4B4843; color: white; font-weight: normal; font-size: 24px;"
    else:
        return "background-color: #4B484340; color: #333333; font-size: 18px;"

def setup_navigation():
    """
    Set up the navigation sidebar for the app.

    Returns:
    list: A list of page names.
    """
    st.sidebar.markdown("<h1 style='text-align: center; color: #AC8D61;'>Navegação</h1>", unsafe_allow_html=True)
    pages = ["Oportunidade de Melhoria", "Refinamento das tarefas", "Planilha Final"]
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    for i, page in enumerate(pages):
        button_class = "current" if i == st.session_state.current_page else "previous" if i == st.session_state.current_page - 1 else ""

        if st.sidebar.button(page, key=f"nav_{i}", use_container_width=True, disabled=(i == st.session_state.current_page)):
            st.session_state.current_page = i
            st.rerun()

        st.markdown(f"""
            <style>
            div.row-widget.stButton > button[key="nav_{i}"] {{
                {get_button_style(button_class)}
            }}
            </style>
            """, unsafe_allow_html=True)

    return pages

def render_oportunidade_melhoria():
    """
    Render the "Oportunidade de Melhoria" page.
    This page includes a form for user input and displays opportunities for improvement.
    """
    st.markdown('<p class="medium-font">Oportunidade de Melhoria</p>', unsafe_allow_html=True)

    # Initialize variables
    file_content = None  
    operaciona_atividade = None
    sistema_relacionado = None
    solucao_gap = None
    outro_gap = None

    # Create a form for user input
    with st.form(key='oportunidade_melhoria_form'):
        # Input fields
        ramo_empresa = st.text_input("Ramo da empresa *", placeholder="Digite o ramo da empresa")
        direcionadores = st.text_input("Direcionadores *", placeholder="Digite os direcionadores de negócios")
        nome_processo = st.text_input("Nome do processo *", placeholder="Digite o nome do processo")
        atividade = st.text_input("Atividade *", placeholder="Digite a atividade")
        evento = st.text_input("Evento *", placeholder="Digite o evento")
        causa = st.text_input("Causa *", placeholder="Digite a causa")
        operaciona_atividade = st.text_input("Quem operacionaliza essa atividade hoje?", placeholder="Digite o responsável pela atividade")
        sistema_relacionado = st.text_input("Possui algum sistema relacionado? Se sim, qual?", placeholder="Descreva o sistema relacionado")
        solucao_gap = st.text_input("Já foi tentada alguma solução para esse gap? Se sim, qual?", placeholder="Descreva a solução tentada")
        outro_gap = st.text_input("Existe algum outro gap relacionado? Se sim, qual?", placeholder="Descreva o outro gap relacionado")
        
        # File uploader for interview transcript
        uploaded_file = st.file_uploader("Anexar transcrição da entrevista", type=["txt"])

        # Display first 3 lines of uploaded file
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8")
            file_lines = file_content.split("\n")
            brief_content = "\n".join(file_lines[:3])
            st.markdown("**Primeiras 3 linhas da Transcrição:**")
            st.code(brief_content, language="text")

        submit_button = st.form_submit_button(label='Buscar Oportunidade de Melhoria')

    # Process form submission
    if submit_button:
        if ramo_empresa and direcionadores and nome_processo and atividade and evento and causa:              
            # Store form data in session state
            processo = st.session_state.processo = [{
                "ramo_empresa": ramo_empresa,
                "direcionadores": direcionadores,
                "nome_processo": nome_processo,
                "atividade": atividade,
                "evento": evento,
                "causa": causa,
                "operaciona_atividade": operaciona_atividade,
                "sistema_relacionado": sistema_relacionado,
                "solucao_gap": solucao_gap,
                "outro_gap": outro_gap,
                "transcrição": file_content if file_content is not None else ""
            }]

            #Apply AI
            with st.spinner('Identificando oportunidade de melhoria...'):  
                resultados = analyze_single_process(
                    process_data=processo,
                    faiss_index_path="process_index.faiss",
                    api_key=api_key
                )
            st.success("Oportunidade de melhoria identificada com sucesso.")

            # Store resultados in the session state
            st.session_state.resultados = resultados

            # Then allow the user to download the results as CSV
            csv = convert_df_to_csv(resultados)
            st.download_button(
                label="Baixar CSV",
                data=csv,
                file_name='oportunidade.csv',
                mime='text/csv',
            )            
        else:
            st.warning("Por favor, preencha todos os campos antes de buscar a oportunidade de melhoria.")

def render_refinamento_tarefas():
    """
    Render the "Refinamento das tarefas" page.
    This page allows users to select and refine improvement opportunities.
    """
    # Retrieve 'resultados' from session state
    if 'resultados' not in st.session_state:
        st.warning("Nenhum resultado disponível. Por favor, volte e busque uma oportunidade de melhoria.")
        return
    
    resultados = st.session_state['resultados']
    lista_resultados = resultados.iloc[:, 0].tolist() 
    
    # If we have stored additional opportunities, add them to the list
    if 'additional_opportunities' not in st.session_state:
        st.session_state.additional_opportunities = []
    
    # Combine original results with additional opportunities
    lista_resultados.extend(st.session_state.additional_opportunities)
    
    st.markdown('<p class="medium-font">Escolha as oportunidades de melhoria:</p>', unsafe_allow_html=True)

    # Display checkboxes for each opportunity
    selected_opportunities = {}
    for i, opportunity in enumerate(lista_resultados):
        key = f"opportunity_{i}"
        selected_opportunities[key] = st.checkbox(opportunity, key=key)

    # Add a new opportunity input
    new_opportunity = st.text_input("Nova oportunidade de melhoria:", key="new_opportunity")
    add_button = st.button("Adicionar Nova Oportunidade")

    if add_button and new_opportunity:
       
        # Update the resultados DataFrame
        new_row = pd.DataFrame({resultados.columns[0]: [new_opportunity]})
        st.session_state.resultados = pd.concat([resultados, new_row], ignore_index=True)
        
        st.success(f"Nova oportunidade adicionada: {new_opportunity}")
        st.rerun()

    # Confirm selection button
    confirm_button = st.button("Confirmar Seleção")

    if confirm_button:
        selected_text = ""
        st.session_state.selected_opportunities = []
        for key, selected in selected_opportunities.items():
            if selected:
                index = int(key.split('_')[1])
                selected_text += f"- {lista_resultados[index]}\n"
                st.session_state.selected_opportunities.append(lista_resultados[index])

        # Store updated selected_text in session state
        st.session_state.selected_text = selected_text

        # Display selected opportunities
        if selected_text:
            st.markdown("**Oportunidades selecionadas:**")
            st.markdown(f'<div class="custom-code-block">{selected_text}</div>', unsafe_allow_html=True)
        else:
            st.info("Nenhuma oportunidade selecionada.")


def render_planilha_final():
    """
    Render the "Planilha Final" page with merged data from selected opportunities and results.
    This page displays a final spreadsheet with editable fields and allows CSV download.
    """
    # Get the selected opportunities from the session state
    selecao = st.session_state.get('selected_opportunities')
    
    # Get the resultados from the session state
    resultados = st.session_state.get('resultados')
    
    # Filter resultados to only include selected opportunities
    if not resultados.empty and selecao:
        # Get the name of the first column (opportunity column)
        opportunity_column = resultados.columns[0]
        
        # Filter resultados where the opportunity column values are in selecao
        filtered_resultados = resultados[resultados[opportunity_column].isin(selecao)].copy()
        
        # Rename columns
        column_mapping = {
            filtered_resultados.columns[0]: 'Oportunidade de Melhoria',
            filtered_resultados.columns[1]: 'Tarefa',
            filtered_resultados.columns[2]: 'Critério de Aceitação'
        }
        filtered_resultados = filtered_resultados.rename(columns=column_mapping)
        
        # Display editable dataframe using st.data_editor
        st.write("Edite os dados conforme necessário:")
        edited_df = st.data_editor(
            filtered_resultados,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Oportunidade de Melhoria": st.column_config.TextColumn(
                    "Oportunidade de Melhoria",
                    width="large",
                    help="Digite a oportunidade de melhoria"
                ),
                "Tarefa": st.column_config.TextColumn(
                    "Tarefa",
                    width="large",
                    help="Digite a tarefa"
                ),
                "Critério de Aceitação": st.column_config.TextColumn(
                    "Critério de Aceitação",
                    width="large",
                    help="Digite o critério de aceitação"
                )
            }
        )
        
        # Update session state with edited data
        st.session_state.final_df = edited_df
        
        # Add download button for filtered CSV
        if not edited_df.empty:
            # Allow the user to download the results as CSV
            csv = convert_df_to_csv(edited_df)
            st.download_button(
                label="Baixar CSV",
                data=csv,
                file_name='oportunidade_melhoria_final.csv',
                mime='text/csv',
            )
    else:
        st.warning("No results to filter or no selections made.")


def main():
    """
    Main function to run the Streamlit app.
    
    This function sets up the page configuration, loads the background and CSS,
    sets up navigation, and renders the appropriate page content based on user navigation.
    It also handles the progress bar and navigation buttons.
    """
    st.set_page_config(page_title="Oportunidade de Melhoria", layout="wide")
    add_bg_from_local('background.png')
    load_css('style.css')  # Load the external CSS file
    
    pages = setup_navigation()
    
    # Calculate progress value
    progress_value = (st.session_state.current_page + 1) / len(pages)
    
    # Create layout with title on the left and logo on the right
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<p class="big-font">{pages[st.session_state.current_page]}</p>', unsafe_allow_html=True)
    with col2:
        st.image('logo.png', width=250)
    
    # Add progress bar
    st.progress(progress_value)

    # Create a container for the main content
    main_container = st.container()
    with main_container:
        # Render appropriate page content based on current page
        if pages[st.session_state.current_page] == "Oportunidade de Melhoria":
            render_oportunidade_melhoria()
        elif pages[st.session_state.current_page] == "Refinamento das tarefas":
            render_refinamento_tarefas()
        elif pages[st.session_state.current_page] == "Planilha Final":
            render_planilha_final()
        else:
            st.write(f"This is {pages[st.session_state.current_page]} page. Add your content here.")

    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.current_page > 0:
            if st.button("Anterior", key="prev_button"):
                st.session_state.current_page -= 1
                st.rerun()
    with col3:
        if st.session_state.current_page < len(pages) - 1:
            if st.button("Próximo", key="next_button"):
                st.session_state.current_page += 1
                st.rerun()
        elif st.session_state.current_page == len(pages) - 1:
            if st.button("Finalizar", key="finish_button"):
                st.success("Processo finalizado com sucesso!")

if __name__ == "__main__":
    main()