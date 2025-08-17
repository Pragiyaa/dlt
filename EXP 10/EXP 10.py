import numpy as np
importtensorflowastf
fromtensorflow.keras.modelsimportModel
fromtensorflow.keras.layersimportInput,LSTM,Embedding,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
fromtensorflow.keras.preprocessing.sequenceimportpad_seque
nces # Sample data
english_sentences=['hello','howareyou','goodmornin
g'] french_sentences = ['bonjour', 'comment ça va',
'bonjour'] #Tokenization
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_sentences)
eng_seq=eng_tokenizer.texts_to_sequences(english_sentences
) eng_word_index = eng_tokenizer.word_index
eng_vocab_size=len(eng_word_index)+
1 fr_tokenizer = Tokenizer(filters='')
fr_tokenizer.fit_on_texts(['<start>'+sent+'<end>'forsentinfrench_sentences])
fr_seq=fr_tokenizer.texts_to_sequences(['<start>'+sent+'<end>'forsentinfrench_sentences])
fr_word_index = fr_tokenizer.word_index
fr_index_word={i:wforw,iinfr_word_index.items()}
fr_vocab_size = len(fr_word_index) + 1
#Pad sequences
max_eng_len=max(len(seq)forseqineng_seq)
max_fr_len = max(len(seq) for seq in fr_seq)
encoder_input_data = pad_sequences(eng_seq, maxlen=max_eng_len, padding='post')
decoder_input_data=pad_sequences([s[:-1]forsinfr_seq],maxlen=max_fr_len-1,padding='post')
decoder_target_data=pad_sequences([s[1:]forsinfr_seq],maxlen=max_fr_len-1,padding='post')
decoder_target_data = np.expand_dims(decoder_target_data, -1)
# Model parameters
embedding_dim=64
latent_dim=128
#Encoder
encoder_inputs=Input(shape=(None,))
enc_emb = Embedding(eng_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm,state_h,state_c=LSTM(latent_dim,return_state=True)(enc_emb)
encoder_states = [state_h, state_c]
#Decoder
decoder_inputs=Input(shape=(None,))
dec_emb=Embedding(fr_vocab_size,embedding_dim)(decoder_inputs)
decoder_lstm,_,_=LSTM(latent_dim,return_sequences=True,return_state=True)(dec_emb,
initial_state=encoder_states)
decoder_dense=Dense(fr_vocab_size,activation='softmax')
decoder_outputs = decoder_dense(decoder_lstm)
#Combinedmodel
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc
uracy']) #Train
model.fit([encoder_input_data,decoder_input_data],decoder_target_data,batch_size=2,epochs=300
, verbose=0)
#Inferencemodels
# Encoder model
encoder_model_inf=Model(encoder_inputs,encoder_sta
tes) # Decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs=[decoder_state_input_h,decoder_state_inpu
t_c]
dec_emb_inf=Embedding(fr_vocab_size,embedding_dim)(decoder_inputs)
decoder_lstm_inf, state_h_inf, state_c_inf = LSTM(
latent_dim,return_sequences=True,return_state=True)(
dec_emb_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf=decoder_dense(decoder_lstm_i
nf) decoder_model_inf = Model(
[decoder_inputs] + decoder_states_inputs,
[decoder_outputs_inf]+decoder_states_inf)
# Translation function
deftranslate(input_tex
t):
input_seq=eng_tokenizer.texts_to_sequences([input_text])
input_seq=pad_sequences(input_seq,maxlen=max_eng_len,padding='po
st') states = encoder_model_inf.predict(input_seq)
target_seq=np.zeros((1,1))
target_seq[0,0]=fr_word_index['<start
>'] stop_condition = False
translated_sentence = []
whilenotstop_condition:
output_tokens,h,c=decoder_model_inf.predict([target_seq]+state
s) sampled_token_index = np.argmax(output_tokens[0, -1, :])
sampled_word = fr_index_word.get(sampled_token_index, '')
ifsampled_word=='<end>'orlen(translated_sentence)>max_fr_len:
stop_condition = True
else:
translated_sentence.append(sampled_word)
target_seq = np.zeros((1, 1))
target_seq[0,0]=sampled_token_ind
ex states = [h, c]
return''.join(translated_sentenc
e) #Example
print("English: how are you")
print("French:",translate("howareyou"
))