# from PyPDF2 import PdfMerger


# def merge_pdf(pdf_list,output_path):
#     pdf_merger = PdfMerger()

#     for pdf in pdf_list:
#         pdf_merger.append(pdf)

#     with open(output_path,'wb') as output_file:
#         pdf_merger.write(output_file)

# pdf_list = [
#     'Ads_Spirits_Pvt_Ltd_vs_Shubhom_Juneja_on_4_May_2023.PDF','Booklet- Laws relating to Women_0.pdf','Bsc_Projects_Private_Limited_vs_Ircon_International_Limited_on_22_March_2023.PDF',
#     'Dinesh_Marwah_vs_State_Of_Nct_Of_Delhi_on_28_March_2023.PDF','Exemption_Yamini_Manohar_vs_T_K_D_Keerthi_on_21_April_2023.PDF','Gautam_Spinners_vs_Commissioner_Of_Customs_Import_New_on_6_January_2023.PDF',
#     'Jumah_Khan_vs_The_State_Govt_Of_Nct_Of_Delhi_on_16_March_2023.PDF','Kewal_Krishan_Kumar_vs_Enforcement_Directorate_on_2_February_2023.PDF','M_S_Gap_International_Sourcing_India_vs_Additional_Commissioner_Cgst_on_1_May_2023.PDF',
#     'Meena_Kumari_vs_Bhagwant_Prasad_Sharma_And_Anr_on_27_February_2023.PDF','Mohd_Imran_Khan_Ors_vs_The_State_Gnct_Of_Delhi_And_Anr_on_7_March_2023.PDF','Ms_M_Prosecutrix_vs_State_Of_Nct_Of_Delhi_Ors_on_5_April_2023.PDF',
#     'Paras_Ram_Dangal_Society_vs_Estate_Officer_Iv_Dda_And_Anr_on_17_February_2023.PDF','Savronik_Sistem_India_Private_Limited_vs_Northern_Railways_Anr_on_23_March_2023.PDF','Sbs_Holding_Inc_vs_Anant_Kumar_Choudhary_Ors_on_7_March_2023.PDF',
#     'Shri_Arun_Kumar_Jain_Anr_vs_1_Shri_Manish_Jaina_Ors_on_14_February_2023.PDF','Star_India_Private_Limited_Anr_vs_Live4Wap_Click_Ors_on_11_January_2023.PDF','Unity_Aurum_Construction_Private_vs_Lok_Sabha_Employees_Cooperative_on_12_April_2023.PDF']

# merge_pdf(pdf_list=pdf_list,output_path="merged_document.pdf")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import PyPDF2
import os

GOOGLE_API_KEY = "AIzaSyCVDoR3U-WDV-_DmeXlt76ubLeNTOw2n64"
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
get_vector_store(get_text_chunks(extract_text_from_pdf("constitution_of_india_full.pdf"))) 
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, if you don't find any answer try to
    change the words to a more formal and creative way that you 
    find the answer from the context, make sure to provide all the details, and don't provide the wrong answer.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9,google_api_key = GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    embedded_question = embeddings.embed_query(user_question)
    docs = new_db.similarity_search_by_vector(embedded_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response

output_dict = user_input("What laws protect me if my husband beats me")
print(output_dict['output_text'])
