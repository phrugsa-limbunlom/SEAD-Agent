class PromptMessage:
    SYSTEM_MESSAGE = """
    You are an AI assistant specialized in analyzing the documents, extracting key insights, and summarizing them.
    Always provide responses that are helpful, polite, and tailored to the user's requirements.
    """
    HUMAN_MESSAGE = """
    Summarize this document
     {query}
    """
    AI_MESSAGE = """ 
    Based on your document, here is the key insights and summarization:

    {context}

    Let me know if you’d like to ask further questions regarding to your document.
    """
    DEFAULT_MESSAGE = """ 
    Hello! I'm an AI assistant specialized in helping users analyze documents, extracting key insights, and summarizing them. 
    I can assist you in summarizing your documents. 
    Please feel free to upload the document, and ask me follow-up questions., 
    and I'll do my best to provide you with meaningful insights.
    """
    FALL_BACK_MESSAGE = """ Sorry, I couldn't find relevant context. Please upload the document first."""
    RELEVANCE_PROMPT = """
        This is prompt template : {template}. Evaluate whether the following query is relevant to the prompt template: {query}. Respond only one word 'relevant' or 'irrelevant'.
    """
    INTENT_PROMPT="""  Classify the user's intent from the input below as one of the following actions:
        - summarize: If the user wants to summarize a paper or document
        - question: If the user is asking a question about a paper
        - fallback: If you are not sure or it doesn't relate to summarization or questions

        Respond with only the word: summarize, question, or fallback

        User Input: {query}
        
        For example
        Summarize the document -> intent = summarize
        Who is the author of the document? -> intent = question
    """