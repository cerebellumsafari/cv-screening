# LLM Demo - CV Screening

A simple demo of how to use the LLM to screen CVs.

User copy/paste a job description and a CV into the text boxes and submit
the formular. The LLM will then extract all requirements from the job,
and make a table of the CV with the requirements as columns along with its
assessment of the candidate.

## Environment variables

| Name                      | Description                  |
|---------------------------|------------------------------|
| ENERGINET_OPENAI_ENDPOINT | Azure OpenAI Endpoint URL    |
| ENERGINET_OPENAI_KEY      | Azure OpenAI Access token    |
| ENERGINET_DEPLOYMENT_NAME | Azure OpenAI Deployment name |
| ENERGINET_API_VERSION     | Azure OpenAI API version     |
