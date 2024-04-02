import os
from langchain_ibm import WatsonxLLM
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
'''
watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = os.getenv('GENAI_VN_EUROPE_APIKEY')
GENAI_VN_EUROPE_PROJECT_ID = os.getenv('GENAI_VN_EUROPE_PROJECT_ID')
'''

params = {
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.DECODING_METHOD: "greedy",
    GenParams.TEMPERATURE: 0,
}
 

llm = WatsonxLLM(
    # model_id = "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    # model_id = "meta-llama/llama-2-13b-chat",
    model_id = "meta-llama/llama-2-70b-chat",
    # model_id = "ibm/granite-13b-chat-v2",
    # model_id = "codellama/codellama-34b-instruct-hf",
    url=os.getenv('GENAI_VN_EUROPE_URL'),
    apikey=os.getenv('GENAI_VN_EUROPE_APIKEY'),
    params = params,
    project_id = os.getenv('GENAI_VN_EUROPE_PROJECT_ID')
)
 
 
#prompt = PromptTemplate.from_template(template)
 
#llm_chain = LLMChain(prompt=prompt, llm=model)
print(llm.invoke("Cual es el mejor amigo de hombre? Por favor, responde siempre en español. RESPUESTA:"))
# q = "What is 1 + 1?"
# generated_response = model.generate(prompt=q)
# print(generated_response['results'][0]['generated_text'])