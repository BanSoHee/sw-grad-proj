from googletrans import Translator

# googletrans
def get_translate(text):

    translator = Translator()
    trans = translator.translate(text, src='ko', dest='en')

    print(f'translation result : {trans.text}')

    return trans.text

# Back Translation
def BT(df):

    return

get_translate('안녕하세요!')