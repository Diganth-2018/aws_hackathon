import streamlit as st
import boto3
import pprint
from botocore.client import Config
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import concurrent.futures

# Initialize the Streamlit app
st.title('Streamlit with AWS Bedrock Example')

# User input for the query
query = st.text_input('Enter your question:')

# Button to submit the query
if st.button('Submit'):
    # AWS Bedrock and other configurations
    kb_id_1 = "H7CMFDJ7YR"
    kb_id = "ZTIEOH1RXQ"
    session = boto3.session.Session()
    region = session.region_name
    bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
    bedrock_client = boto3.client('bedrock-runtime', region_name=region)
    bedrock_agent_client = boto3.client("bedrock-agent-runtime", config=bedrock_config, region_name=region)

    def retrieve(query, kbId, numberOfResults):
        return bedrock_agent_client.retrieve(
            retrievalQuery={
                'text': query
            },
            knowledgeBaseId=kbId,
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': numberOfResults
                }
            }
        )

    def retrieve_information(query, kb_id, num_results):
        response = retrieve(query, kb_id, num_results)
        return response['retrievalResults']

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(retrieve_information, query, kb_id_1, 10)
        future2 = executor.submit(retrieve_information, query, kb_id, 10)
        retrievalResults_1 = future1.result()
        retrievalResults_2 = future2.result()
    
    def reciprocal_rank_fusion(*ranked_lists, k=60):
        scores = {}
        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                doc = result['content']['text']
                scores[doc] = scores.get(doc, 0) + 1 / (rank + k)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs]
    
    fused_results = reciprocal_rank_fusion(retrievalResults_1, retrievalResults_2)
    
    def augment_query(query, documents):
        fused_context = " ".join(documents)
        augmented_query = f"{query} Context: {fused_context}"
        return augmented_query
    
    augmented_query = augment_query(query, fused_results)
    
    bedrock = boto3.client(service_name="bedrock-runtime")
    llm = BedrockChat(model_id="anthropic.claude-3-opus-20240229-v1:0", client=bedrock, model_kwargs={"max_tokens": 1000})

    # Define generic prompt templates for different branches
    identify_topic_prompt = PromptTemplate(
        input_variables=["context"], 
        template="Step 1: Based on the provided context, identify the main topic. If the context is insufficient, state 'I don't know'.\nContext: {context}"
    )

    detailed_explanation_prompt = PromptTemplate(
        input_variables=["topic", "context"], 
        template="Step 2a: Provide a detailed explanation for the identified topic: {topic}. Context: {context}. If the context is insufficient, state 'I don't know'."
    )

    polished_language_prompt = PromptTemplate(
        input_variables=["topic", "context"], 
        template="Step 2b: Provide a polished explanation for the identified topic: {topic}. Context: {context}. If the context is insufficient, state 'I don't know'."
    )

    summarize_prompt = PromptTemplate(
        input_variables=["detail", "context"], 
        template="Step 3: Summarize the key points from the detail provided: {detail}. Context: {context}. If the context is insufficient, state 'I don't know'."
    )

    evaluate_prompt = PromptTemplate(
        input_variables=["query", "context", "result1", "result2"],
        template="Given the query '{query}' and context '{context}', evaluate which of the two results is better:\nResult 1: {result1}\nResult 2: {result2}\nSelect the better result. Do not select 'I don't know' responses."
    )

    # Create chains for each step
    identify_topic_chain = LLMChain(llm=llm, prompt=identify_topic_prompt)
    detailed_explanation_chain = LLMChain(llm=llm, prompt=detailed_explanation_prompt)
    polished_language_chain = LLMChain(llm=llm, prompt=polished_language_prompt)
    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
    evaluate_chain = LLMChain(llm=llm, prompt=evaluate_prompt)

    # Function to evaluate results
    def evaluate_results(query, context, result1, result2):
        evaluation_result = evaluate_chain.run({
            "query": query,
            "context": context,
            "result1": result1,
            "result2": result2
        })
        return evaluation_result

    # Function to run the tree of thought reasoning
    def run_tree_of_thought(query, context):
        # Step 1: Identify the main topic
        topic_result = identify_topic_chain.run({"context": context})

        # Step 2a: Provide a detailed explanation
        detailed_explanation_result = detailed_explanation_chain.run({"topic": topic_result, "context": context})

        # Step 2b: Provide a polished explanation
        polished_language_result = polished_language_chain.run({"topic": topic_result, "context": context})

        # Ensure we don't use "I don't know" responses
        if "I don't know" in detailed_explanation_result:
            best_result = polished_language_result
        elif "I don't know" in polished_language_result:
            best_result = detailed_explanation_result
        else:
            # Evaluate which detail is better
            best_result = evaluate_results(query, context, detailed_explanation_result, polished_language_result)

        # Step 3: Summarize the best detail
        summary_result = summarize_chain.run({"detail": best_result, "context": context})

        return summary_result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_result = executor.submit(run_tree_of_thought, query, augmented_query)
        result = future_result.result()
    
    # Display the results
    st.write('Results:')
    st.write(result)
