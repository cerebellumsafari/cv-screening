from typing import Tuple

import streamlit as st
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
    Below is a list of requirements for a job along with a personal CV.
    For each requirement in the list, asses whether or not the
    candidate is a good match.

     Output a table in markdown format with 3 columns:
     1) The requirement itself
     2) Your assessment whether the candidate fulfills the requirement (YES/NO)
     3) An explanation of your assessment


    ##### JOB REQUIREMENTS START #####
    {job_requirements}
    ##### JOB REQUIREMENTS END #####


    ##### CANDIDATE CV START #####
    {cv_text}
    ##### CANDIDATE CV END #####
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


# -- Streamlit app ------------------------------------------------------------


st.set_page_config(
    page_title='CV Screening',
    page_icon=":bird:",
)

st.header('CV Screening')

job_requirements = None
screening_table = None

with st.form('screening'):
    job_text = st.text_area('Enter job description in plain text:')
    cv_text = st.text_area('CV in plain text:')

    if st.form_submit_button('Start Screening'):
        job_requirements, screening_table = invoke_chain(
            job_text=job_text,
            cv_text=cv_text,
        )

if job_requirements:
    st.subheader('Job requirements')
    st.markdown(job_requirements)
if screening_table:
    st.subheader('Screening result')
    st.markdown(screening_table)
