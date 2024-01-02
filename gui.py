import fitz
import streamlit as st
from typing import Tuple
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

from config import (
    ENERGINET_OPENAI_ENDPOINT,
    ENERGINET_DEPLOYMENT_NAME,
    ENERGINET_API_VERSION,
    ENERGINET_OPENAI_KEY,
)


# -- Langchain ----------------------------------------------------------------


llm = AzureChatOpenAI(
    azure_endpoint=ENERGINET_OPENAI_ENDPOINT,
    deployment_name=ENERGINET_DEPLOYMENT_NAME,
    openai_api_version=ENERGINET_API_VERSION,
    openai_api_key=ENERGINET_OPENAI_KEY,
    # temperature=0,
)

job_requirements_template = PromptTemplate.from_template("""
    Generate a list of requirements from the job below.
    Output the list in markdown format without headlines:

    {job_text}
""")

cv_screening_template = PromptTemplate.from_template("""
    Below is a list of requirements for a job along with one or more personal CVs.
    For each requirement in the list, asses whether or not each of the
    candidates are a good match.

     Output a table in markdown format with these columns:
     - A column with the requirement itself
     - For each candidate, one column with the assessment (YES/NO)
     - For each candidate, one column with an explanation of the assessment


    ##### JOB REQUIREMENTS START #####
    {job_requirements}
    ##### JOB REQUIREMENTS END #####


    {cv_text}
""")


def invoke_chain(job_text: str, cv_text: str) -> Tuple[str, str]:
    """
    Invoke the langchain chain.
    Passes cv_text and job_text to the chain, and returns the chain output.
    Returns the parsed job requirements along with the screening result
    as a table in markdown format.

    :param job_text: Job description in plain text.
    :param cv_text: CV in plain text.
    :return: Tuple of (job_requirements, cv_screening)
    """
    job_requirements_chain = (
             job_requirements_template
             | llm
             | StrOutputParser())

    cv_screening_chain = (
            cv_screening_template.partial(cv_text=cv_text)
            | llm
            | StrOutputParser())

    chain = (
            {'job_requirements': job_requirements_chain}
            | RunnablePassthrough.assign(screening=cv_screening_chain))

    output = chain.invoke({'job_text': job_text})

    return output['job_requirements'], output['screening']


# -- PDF files ----------------------------------------------------------------


def extract_text_from_pdf(pdf_file: bytes) -> str:
    """
    Extract text from a PDF file.

    :param pdf_file: PDF file as bytes.
    :return: Text from PDF file.
    """
    text = ''
    with fitz.open(stream=pdf_file, filetype='pdf') as doc:
        for page in doc:
            text += page.get_text()
    return text


# -- Streamlit app ------------------------------------------------------------


st.set_page_config(
    page_title='CV Screening',
    page_icon=":bird:",
)

st.header('CV Screening')

cv_texts = []
job_requirements = None
screening_table = None

with st.form('screening'):
    job_text = st.text_area('Enter job description in plain text:')
    cv_text = st.text_area('CV in plain text:')
    uploaded_files = st.file_uploader(
        label='Or upload CVs as PDF files (up to 3):',
        type=['pdf'],
        accept_multiple_files=True,
    )

    if cv_text:
        cv_texts.append(cv_text)

    for uploaded_file in uploaded_files:
        cv_bytes_data = uploaded_file.read()
        cv_text_data = extract_text_from_pdf(cv_bytes_data)
        cv_texts.append(cv_text_data)

    cv_text_concat = '\n\n'.join([
        f'##### CV {i} START #####\n{cv_text}\n##### CV {i} END #####'
        for i, cv_text in enumerate(cv_texts)
    ])

    if st.form_submit_button('Start Screening'):
        job_requirements, screening_table = invoke_chain(
            job_text=job_text,
            cv_text=cv_text_concat,
        )

if job_requirements:
    st.subheader('Job requirements')
    st.markdown(job_requirements)
if screening_table:
    st.subheader('Screening result')
    st.markdown(screening_table)
