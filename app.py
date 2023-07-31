import os
from datetime import time

import openai
import pandas as pd
import streamlit as st
from llama_index import (KeywordTableIndex, LLMPredictor, ServiceContext,
                         SimpleDirectoryReader, StorageContext,
                         VectorStoreIndex, load_index_from_storage)
from llama_index.llms import OpenAI
from requests.exceptions import RequestException
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from streamlit_chat import message
from youtube_transcript_api import YouTubeTranscriptApi

from src.cache import cache_get, cache_set

# get openai api key from secrets tab
key = st.secrets["openai_key"]
# Set up OpenAI API credentials
openai.api_key = key

# set the page layout to wide
st.set_page_config(layout="wide")

# devide the page into two columns
slide1, slide2 = st.columns( [ 0.8 , 0.6 ] )


def load_video_content( video_id ):
    slide2.video( f'https://youtu.be/{video_id}' )


def convert_single_video(video_id):

    print( "loading" )

    try:
        # validate cache on given video id / if that canbe found inside the cache then return it
        cache = cache_get(video_id)
        if cache is not None:
            
            # if the id inside the cache then we can load the index from the storage
            storage_context = StorageContext.from_defaults(persist_dir="storage")
            # laod index
            index = load_index_from_storage(storage_context, index_id= f"index_{video_id}")

            # configure the query engine with customised parameters
            query_engine =  index.as_query_engine(
                                similarity_top_k=1,
                                vector_store_query_mode="default"
                            )
            st.session_state.query_engine = query_engine

        # fetach youtube provided transcript or generate using whisper model
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            text = " ".join([ line['text'] + f'( time: {line["start"]} )' for line in transcript])
            cache_set( video_id , text )

            time_stamp = [ { "time" : line['start'] , 'text': line['text'] }  for line in transcript ]
            # convert the data into dataframe object
            df_transcript = pd.DataFrame( time_stamp )
            print(df_transcript)
            # set the state variable to the transcript
            st.session_state.transcript = df_transcript

            # write this text to a folder
            if not( os.path.exists( os.path.join('data', video_id ) ) ):
                # create the directory
                os.makedirs( os.path.join( 'data' , video_id ) )

            # write data to a text file
            with open( os.path.join( 'data' , video_id , 'trancript.txt' ) , 'w' ) as f:
                # write data tp the file
                f.write( text )
                f.close()

            # index the data using llamaindex 
            """
            use simpledirectory reader and vectorstoreinex method , try out with different index types
            """
            # load docs using directory reader
            docs = SimpleDirectoryReader( os.path.join( 'data' , video_id ) ).load_data()

            # define the LLM
            gpt_llm = OpenAI(temperature=0.1, 
                         model="text-davinci-003", 
                         max_tokens=512,
                         top_p=0.8,
                         frequency_penalty=0.0,
                         presence_penalty=0.6)
            
            service_context = ServiceContext.from_defaults(llm=gpt_llm)

            # generate the index from documents ( we can save index based on requirement )
            """
            vector stores :  simple in memeory store , Faiss , weaviate , Pinecone , Chorma
            """
            index = VectorStoreIndex.from_documents( docs ,  service_context = service_context )

            # save the index into storage
            index.set_index_id( f'index_{video_id}' )
            index.storage_context.persist( "storage/" )

            # configure the query engine with customised parameters
            query_engine =  index.as_query_engine(
                                similarity_top_k=1,
                                vector_store_query_mode="default"
                            )


            st.session_state.query_engine = query_engine


        except RequestException as e :
            st.error(f"Error retrieving transcript for Video ID: {video_id}")
        
    except RequestException as e:
        st.error(f"Error converting video with Video ID: {video_id}")
        st.error(str(e))


def init_session():

    print("initializing")

    # define the streamlit session states
    if "video_id" not in st.session_state:
        st.session_state.video_id = ''
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ''
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [ ] 
    if "human_prompt" not in st.session_state:
        st.session_state.human_prompt = '' 
    if "index" not in st.session_state :
        st.session_state.index = False
    if "transcript" not in st.session_state :
        st.session_state.transcript = pd.DataFrame()
    if "chat_status" not in st.session_state:
        st.session_state.chat_status = ''
    if "blog_status" not in st.session_state:
        st.session_state.blog_status = ''
    if "prompt" not in st.session_state :
        st.session_state.chat_prompt = """
                                        Please answer the question below based solely on the context information provided.
                                        You must provide an answer in one paragraph.
                                        If you have a question unrelated that can not be answered with the provided context, please reply with \"Out Of Context Question\" as your response.\n
                                        """
        st.session_state.time_extractor = "Provide the time which can refer to answer the query.Don't show time's everywhere inside the text only display the first time instance in seperate new line ."
        st.session_state.seo_prompt = "Transform this given context and user instruction into an SEO blog post also make it intuitive inspiring and captivating also make it a little bit on the longer \
                                            side while maintaining SEO friendliness at the highest level with a title , make it interesting to read and easy to understand , \
                                            strictly follow the given instructions ." 

    


def on_click_callback():
    # user input
    human_prompt = st.session_state.human_prompt

    # llm output generation
    gpt_query =  st.session_state.chat_prompt + human_prompt + st.session_state.time_extractor
    llm_response = str( st.session_state.query_engine.query( gpt_query ) )

    # add to the chat history
    st.session_state.chat_history.append(
            { 'human' : human_prompt , 'ai' : llm_response } 
        )
    


def seo_callback(seo_query):
    # seo user query
    seo_context =  st.session_state.seo_prompt + seo_query 
    print(seo_context)

    # generate openai completion output
    response =  openai.Completion.create(
                    engine="text-davinci-003",
                    prompt= seo_context ,
                    temperature=0.7,
                    max_tokens=500,
                    top_p=0.8,
                    frequency_penalty=0.0,
                    presence_penalty=0.6,
                    stop=None )
    
    blog_post = response.choices[0].text.strip()

    return blog_post
    


def main():
    
    menu = ['Home', 'About us' ]
    choice = st.sidebar.selectbox('Navigation', menu)
    
    if choice == 'Home':

        with slide1 :
            st.title('Tuber - Guru🚀🚀🚀')

            st.markdown('## Convert Single Video')
            video_id = st.text_input('YouTube Video ID')

            # set the video id to a session state
            st.session_state.video_id =  video_id
            if st.button('Indexing'):
                with st.spinner('index generation...'):
                    convert_single_video(video_id)

            st.markdown('## Chat Arena:')

            with st.form(key='chat',clear_on_submit=True)  :
                st.text_input("Hi There! Enter your question" , value= "" , key = "human_prompt")

                submit = st.form_submit_button(label='Chat', on_click= on_click_callback )
                with st.spinner('Working on it...'):
                    for i in sorted( range( len(st.session_state.chat_history)  ) , reverse=True ) :     
                        message(
                            st.session_state.chat_history[i]['ai'], 
                            allow_html=True ,
                            key = f'{i}_ai'
                        )

                        message( st.session_state.chat_history[i]['human']  , is_user=True , key = f'{i}_human' )
                            

                    
        with slide2 :
            video_id = ''
            load_video_content( st.session_state.video_id )

            with st.container():
                st.markdown("## Generate Blog Post")
                slider =  st.slider( "select video content time span :",
                                min_value = 0.0, 
                                max_value = 1200.0, 
                                value=(0.0, 120.0)
                                )

                st.text_input(
                                "Input user requirements",
                                key="seo_query",
                            )

                if st.button('Generate'):
                        st.markdown('## Generated Blog Post:')
                        t1 , t2 = slider[0] , slider[1]
                        #filter the dataframe with transcript
                        df = st.session_state.transcript
                        df_tmp = df[ ( df.time >= t1 ) & ( df.time <= t2 ) ]
                        seo_context = ' '.join(df_tmp['text'].tolist())
                        with st.spinner('index generation...'):
                            blog_post =  seo_callback( seo_query= seo_context )
                            st.markdown( f'### generated blog: \n\n{blog_post}' )


if __name__ == '__main__':
    init_session()
    main()

