import os
import openai


openai.api_key = os.environ.get('OPENAI_KEY')
completion = openai.ChatCompletion()



def process_question(question, context):
    query = f"""Given the context:
    {context}
    Answer this question only using the context above
    {question}
    """

    return query


def query(question, chat_log=None):
    if chat_log is None:
        chat_log = [{
            'role': 'system',
            'content': 'You are a helpful, upbeat and funny assistant.',
        }]
    chat_log.append({'role': 'user', 'content': question})
    response = completion.create(model='gpt-3.5-turbo', messages=chat_log)
    answer = response.choices[0]['message']['content']
    chat_log.append({'role': 'assistant', 'content': answer})
    return answer