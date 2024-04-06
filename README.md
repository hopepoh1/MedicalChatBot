End-to-end-Medical-Chatbot-using-Llama2

Bu bot medikal bilgilere sahip 700 sayfalık bir pdf'i kullanarak  kullanıcının sorularına cevap veriyor.

Usage of this bot is after giving a pdf about medical stuff,user can ask anything and bot answer it from the pdf we given.

## How to run?
STEPS:

 Repsitory'yi kopyala.

 First Clone the repository.

Project repo: https://github.com/hopepoh1/MedicalChatBot.git

STEP 01- İsteğinize göre bir envoriment oluşturun.

Create a conda environment after opening the repository.

conda create -n medicalchatbot python=3.9 -y
conda activate medicalchatbot

STEP 02-gerekli  kütüphaneleri indirelim.

install the requirements.

pip install -r requirements.txt

.env dosyası oluşturalım. içine bu projede lazım olan pinecone keyinizi yazın.

Create a .env file in the root directory and add your Pinecone credentials as follows:

PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

projede kullanıcağımız LLama modelini indirelim.

Download the quantize model from the link provided in model folder & keep the model in the model directory:
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
# run the following command
python store_index.py(biraz uzun sürecek çünkü datayı pinecone indexine kopyalıyoruz.)(its gonna take a while)

# Finally run the following command
python app.py

open up localhost:

*******************
Techstack Used:

ctransformers
sentence-transformers

pinecone-client

langchain

flask

pypdf

langchain_pinecone

python-dotenv

-e .
