from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

from langchain.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign

from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain.vectorstores import FAISS

from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

from operator import itemgetter

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)


instruct_llm = ChatOpenAI(
  openai_api_key="<REPLACE WITH YOUR OPEN-ROUTER API KEY",
  openai_api_base="https://openrouter.ai/api/v1",
  model_name="mistralai/mistral-7b-instruct:free"
)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)

embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

conversation = [  ## This conversation was generated partially by an AI system, and modified to exhibit desirable properties
    "[User]  This course is about Electricity and Magnetism",
    "[Agent] Electrostatics; Electric Fields; Gauss's Law; Electric Potential; Capacitors and Dielectrics; Current and Resistance; Magnetic Fields and Ampere's Law",
    "[User] Here is my course PDF for Multivariable Calculus",
    "[Agent] Limits and Partial Derivatives; Tangent Planes and Linear Approxiamation; Directional Derivatives and Gradient Vectors; Double Integrals; Triple Integrals; Taylor Polynomials and Polynomial Interpolation; Infinite and Power Series"
    "[User] Textbook for Mechanics"
    "[Agent] Kinematics; Newton's Laws; Work and Energy; Momentum and Impulse; Rotational Motion; Gravitation; Oscillations and Waves" 
]

convstore = FAISS.from_texts(conversation, embedding=embedder)


def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

class question_generator:

    def __init__(self, file_name, description):
        self.description = description

        doc = UnstructuredFileLoader(file_name).load()
        content = json.dumps(doc[0].page_content)
        docs_chunks = [text_splitter.split_documents(doc)]
        docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]

        doc_string = "Available Documents:"
        doc_metadata = []

        for chunks in docs_chunks:
            metadata = getattr(chunks[0], 'metadata', {})
            print('METADATA: ')
            print(metadata)
            doc_string += "\n - " + (metadata.get('source').split('.')[0])
            doc_metadata += [str(metadata)]
    
        extra_chunks = [doc_string] + doc_metadata


        vecstores = [FAISS.from_texts(extra_chunks, embedder)]
        vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]

        self.docstore = aggregate_vstores(vecstores)

        initial_msg = (
        "Hello! I am a document chat agent here to help the user!"
        f" I have access to the following documents: {doc_string}\n\nHow can I help you?"
        )

        





    def topics_generator(self):
        chat_prompt = ChatPromptTemplate.from_messages([("system",
        "You are an academic TOPIC creator chatbot who divides the course into 7 TOPICS"
        "This is information about the user's course: {input}\n\n"
        " From this, we have retrieved the following potentially-useful info: "
        "Conversation History Retrieval:\n{history}\n\n"
        " Document Retrieval:\n{context}\n\n"
        " (Make sure the topics are only from the sources being used). But don't site the sources"
        "There must be ONLY 7 TOPIC words seperated by semicolons covering the whole course in only one line"
        "No bullet points, lists, paragraphs, sentences, seperate lines, brackets, parentheses, or phrases"
        "Only include the big topics which are course units. Ensure that the topics are in sequential order from least to most difficult"
        "Example Output: 'Kinematics; Newton's Laws; Work and Energy; Momentum and Impulse; Rotational Motion; Gravitation; Oscillations and Waves'"
    ), ('user', '{input}')])
        
        stream_chain = chat_prompt| RPrint() | instruct_llm | StrOutputParser()

        long_reorder = RunnableLambda(LongContextReorder().transform_documents)

        retrieval_chain = (
        {'input' : (lambda x: x)}
    ## TODO: Make sure to retrieve history & context from convstore & docstore, respectively.
    ## HINT: Our solution uses RunnableAssign, itemgetter, long_reorder, and docs2str
        | RunnableAssign({'history' : itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
        | RunnableAssign({'context' : itemgetter('input') | self.docstore.as_retriever()  | long_reorder | docs2str})
        | RPrint()
        )

        def chat_gen(message, history=[], return_buffer=True):
            buffer = ""
    ## First perform the retrieval based on the input message
            retrieval = retrieval_chain.invoke(message)
            line_buffer = ""

    ## Then, stream the results of the stream_chain
            for token in stream_chain.stream(retrieval):
                buffer += token
        ## If you're using standard print, keep line from getting too long
                yield buffer if return_buffer else token

    ## Lastly, save the chat exchange to the conversation memory buffer
            save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)
        
        test_question = self.description  ## <- modify as desired

        topics = ""

        for response in chat_gen(test_question, return_buffer=False):
            topics += response
        
        topic_list = topics.split(';')
        print(topic_list, len(topic_list))

topics = question_generator(file_name='ElectricityAndMagnetism.pdf', description = "My course's PDF electricity and magnetism")
topics.topics_generator()

