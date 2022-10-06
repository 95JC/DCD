#!/usr/bin/env python
# coding: utf-8

# # Introducción al NLTK
# El kit de herramientas de lenguaje natural (<em>Natural Language Toolkit</em>), es un conjunto de bibliotecas y programas para el procesamiento del lenguaje natural simbólico y estadísticos para el lenguaje de programación Python. NLTK incluye demostraciones gráficas y datos de muestra.
# 
# ### Instalación y descarga de datos
# conda install -c anaconda nltk
# 
# ![imagen.png](attachment:imagen.png)

# In[2]:


import nltk
nltk.download()

# Seleccionar la colección Book y descargarla


# In[3]:


# http://www.nltk.org/book/ch01.html
# Cargar el paquete usando el interprete
from nltk.book import *


# In[4]:


# Para ver algún texto, solo ingresar el nombre en el interprete
text1


# In[5]:


text2


# In[6]:


' '.join(text1.tokens)


# ### Procesamiento del Lenguaje

# In[7]:


# Buscar una palabra en un texto:
text1.concordance("monstrous")


# <p><b>Nota:</b> Intente realizar otras búsquedas, por ejemplo: en <em>Sense and Sensibility</em> la palabra afecto (<em>affection</em>) usando: <code>text2.concordance("affection")</code>. Busque en el libro <em>Genesis</em> para averiguar cuánto tiempo vivieron algunas personas, usando <code>text3.concordance("lived")</code>. Puede consultar text4: <em>Inaugural Address Corpus</em>, para ver ejemplos de inglés que se remontan a 1789, y buscar palabras como <em>nation, terror, god</em> para ver cómo estas palabras se han utilizado de manera diferente a lo largo del tiempo. En el <em>texto5: Chat Corpus:</em> busque en estas palabras poco convencionales como <em>im, ur, lol</em>.</p>

# In[8]:


# Buscar palabras similares, de acuerdo con similar
text1.similar("monstrous")


# In[12]:


text2.similar("monstrous")


# In[14]:


# El método common_context permite examinar el contexto compartido por 2 o más palabras
text2.common_contexts(["monstrous", "very"])


# In[15]:


# Obtener la longitud de un texto
len(text3)


# In[18]:


# Calcular el vocabulario de un texto
sorted(set(text3))


# In[19]:


len(set(text3))


# In[21]:


# Riqueza léxica de un texto
len(set(text3)) / len(text3)
'''
La riqueza léxica es la relación que existe entre la extensión de un
texto y el número de palabras distintas que contiene.
'''


# In[24]:


# Calcular frecuencia de palabra en texto
print( text3.count("smote") )
print( 100 * text4.count('a') / len(text4) )


# ¿Cuántas veces aparece la palabra <em>lol</em> en text5? ¿Cuánto es esto como porcentaje del número total de palabras en este texto? 

# In[25]:


print( 'La palabra lol en el text5 aparece:', text5.count("lol"), 'veces' )
print( 'Porcentaje de su ocurrencia:', 100 * text5.count('lol') / len(text5) )


# <a href="http://www.nltk.org/book/ch02.html">Accessing Text Corpora and Lexical Resources</a>
# ### Acceso a corpora de texto y recursos léxicos
# <ul>
#     <li>Uso de corpora en procesamiento de lenguaje natural</li>
#     <li>Acceso a recursos léxicos en Python</li>
#     <li>Acceso a corpora de texto en Python</li>
# </ul>
# 
# <b>Corpus de Gutenberg</b>
# <p>NLTK incluye una pequeña selección de textos del archivo de texto electrónico del Proyecto Gutenberg, que contiene unos 25 000 libros electrónicos gratuitos, alojados en http://www.gutenberg.org/.</p>

# In[28]:


# import nltk
# nltk.corpus.gutenberg.fileids()

from nltk.corpus import gutenberg
gutenberg.fileids()


# In[10]:


# Escojamos el primero de estos textos, Emma de Jane Austen
emma = gutenberg.words('austen-emma.txt')
len(emma)


# SInopsis: Emma se enfrenta a un vacío en su vida y con un gran dilema: cómo ayudar a los demás a tener una vida tan perfecta como la suya. En contra del consejo de Knightley, busca posibles novios para su nueva amiga, Harriet Smith, una joven sencilla y modesta, alejada del estilo de vida de la alta sociedad.
# 
# Texto original: Emma (1816) en Wikisource
# 
# Fecha de publicación: 23 de diciembre de 1815
# 
# Género: Novela

# In[11]:


for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid)) # número de caracteres
    num_words = len(gutenberg.words(fileid)) # número de palabras
    num_sents = len(gutenberg.sents(fileid)) # número de oraciones
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid))) # número de vocabulario
    
    # Porcentaje de número de caracteres por palabras, palabras por oraciones y palabras por vocabulario
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)


# <p>La función <b>raw()</b> nos da el contenido del archivo sin ningún procesamiento lingüístico. Entonces, por ejemplo, <b>len (gutenberg.raw ('blake-poems.txt'))</b> nos dice cuántas letras aparecen en el texto, incluidos los espacios entre palabras.</p>
#     
# <p>La función <b>sents()</b> divide el texto en sus oraciones, donde cada oración es una lista de palabras</p>
# 
# <p>- Otros corpora en NLTK: Web and Chat Text, Brown Corpus, Reuters Corpus, Inaugural Address Corpus, Annotated Text Corpora. Pueden ver los ejemplos <a href="http://www.nltk.org/book/ch02.html">aquí</a>.</p>

# ## Recursos léxicos

# In[18]:


# Lista de palabras en inglés: (utilizado por algunos correctores ortográficos https://en.wikipedia.org/wiki/Words_(Unix))
from nltk.corpus import words
print( len(words.words()) )
print(words.words()[800:900])


# In[36]:


"fine" in words.words()


# In[43]:


# Lista de palabras cerradas, palabras con frecuencia alta
# Suelen tener poco contenido léxico y su presencia en un texto no lo distingue de otros textos.
# Palabras vacías: nombre que reciben las palabras sin significado como artículos, pronombres, preposiciones, etc.
# que son filtradas antes o después del procesamiento de datos en lenguaje natural.

from nltk.corpus import stopwords
stopwords.words('english') # también disponible para el español - 'spanish'


# <b>Ejercicio:</b> Definir una función para calcular qué fracción de palabras en un texto no están en la lista de palabras vacías

# In[2]:


def content_fraction(text):
    ''''''

content_fraction(text1) # text1: Moby Dick by Herman Melville 1851


# NLTK incluye el Diccionario de pronunciación CMU para inglés estadounidense, que fue diseñado para su uso por sintetizadores de voz. Para cada palabra, este léxico proporciona una lista de códigos fonéticos (etiquetas distintas para cada sonido contrastivo) conocidos como teléfonos. Observe que fire tiene dos pronunciaciones (en inglés de EE. UU.): F AY1 R de una sílaba y F AY1 ER0 de dos sílabas. Los símbolos en el Diccionario de pronunciación de CMU son del Arpabet, que se describen con más detalle en http://en.wikipedia.org/wiki/Arpabet
# 

# In[48]:


# Diccionario de pronunciación
entries = nltk.corpus.cmudict.entries()
print(len(entries))
for entry in entries[42371:42379]:
    print(entry)


# Lista de palabras comparativas

# In[49]:


from nltk.corpus import swadesh
swadesh.fileids()


# In[54]:


es2en = swadesh.entries(['es', 'en'])
print(es2en)


# In[ ]:





# ## Recursos léxicos
# - Wordnet (http://www.nltk.org/howto/wordnet.html)
# <p>Es una enorme base de datos léxica. Wordnet agrupa las palabras en conjuntos de sinónimos llamados synsets, los cuales
# proporcionan definiciones generales, almacenando las relaciones semánticas entre estos conjuntos de sinónimos.</p>
# 
# ![imagen.png](attachment:imagen.png)
# 
# <em>Wordnet: en otros idiomas:</em>
# <ul>
#     <li><a href="https://github.com/pln-fing-udelar/wnmcr-transform">WN-MCR-Transform</a></li>
#     <li><a href="http://adimen.si.ehu.es/web/MCR/">Multilingual Central Repository</a></li>
# <ul>

# In[56]:


from nltk.corpus import wordnet as wn
wn.synsets('motorcar') # indica synsets de la palabra 'motorcar'


# In[57]:


wn.synset('car.n.01').lemmas() # diferentes tipos de lemma de esa palabra


# In[58]:


[str(lemma.name()) for lemma in wn.synset('car.n.01').lemmas()] # extraer los lemmas en una lista


# In[59]:


wn.synset('car.n.01').definition() # obtiene la definicion


# In[ ]:





# ### Procesamiento de texto plano y estructurado
# - Escribir programas para acceder a archivos locales
# - Dividir documentos en palabras, símbolos de puntuación y realizar análisis
# - Producir salidas con formato en un archivo

# In[69]:


# Acceder a archivos locales
f = open('moby.txt')
raw = f.read()

with open('moby.txt', 'r') as f:
    moby_raw = f.read()

# Leer un archivo línea por línea
f = open('moby.txt', 'r')
for line in f:
    print(line.strip())
    
# Podemos usar la novela como texto plano ya cargada o desde el módulo nltk.book que sería el text1


# In[68]:


# Tokenizar un texto, produce una lista de palabras y puntuaciones
tokens = nltk.word_tokenize(moby_raw)
print( type(tokens), len(tokens) )
print( tokens[:10] )

# Encontrar la posición de una cadena en el texto
print( moby_raw.find("CAPTAIN"), moby_raw.rfind("CAPTAIN") )


# ### Extracción de raíces y lematización

# In[76]:


# Usemos estos 2 textos para ver las diferentes formas de la misma palabra
input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ')
print(words1)

udhr = nltk.corpus.udhr.words('English-Latin1')
print(udhr[:20])


# In[74]:


# Encontrar raíz de palabra
porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1]


# In[75]:


[porter.stem(t) for t in udhr[:20]]


# In[77]:


# Lematización
WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in udhr[:20]]


# ### Dividir en oraciones
# ¿Cómo dividirías oraciones de una cadena de texto larga?

# In[78]:


texto = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is! "
sentences = nltk.sent_tokenize(texto)
print( len(sentences) )
print( sentences )


# ### Categorización y etiquetado de palabras
# - Definición y uso de categorías léxicas
# - Estructuras de datos para almacenamiento de palabras y categorías en Python
# - Etiquetamiento automático de palabras
# 
# Etiquetado de Parte de Oración (Etiquetado Penn Treebank)
# ![imagen.png](attachment:imagen.png)

# In[79]:


nltk.help.upenn_tagset('MD')


# In[80]:


# Identificar la etiqueta POS de una lista de palabras
text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)


# In[84]:


# Ambigüedad
text2 = nltk.word_tokenize("Visiting aunts can be a nuisance")
nltk.pos_tag(text2)

# Otra alternativa de etiquetado POS
# [('Visiting’, 'JJ'), ('aunts', 'NNS'), ('can', 'MD'), ('be', 'VB'), ('a', 'DT'), ('nuisance', 'NN')]


# In[85]:


# Un token etiquetado en NLTK se representa como una tupla
tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token


# In[86]:


# Leer texto etiquetado en NLTK
nltk.corpus.brown.tagged_words()


# In[87]:


nltk.corpus.brown.tagged_words(tagset='universal')


# In[88]:


# Conjunto de etiquetas universal
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.most_common()


# In[ ]:





# ## Spacy
# - Instalación
# <ul>
#     <li>conda install -c conda-forge spacy</li>
#     <li>python -m spacy download es</li>
#     <li>python -m spacy download es_core_news_md</li>
# </ul>
# - Documentación
# <ul>
#     <li><a href="https://spacy.io/usage">Spacy</a></li>
#     <li><a href="https://nlpforhackers.io/complete-guide-to-spacy/">Complete Guide to spaCy</a></li>
# </ul>

# In[90]:


import spacy
from spacy.lang.es.examples import sentences

nlp = spacy.load('es_core_news_sm')
doc = nlp(sentences[0])
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)


# ## Freeling
# - Documentación http://nlp.lsi.upc.edu/freeling/index.php/node/1
# - Los idiomas soportados actualmente son inglés, español, portugués, francés, italiano, alemán, ruso, noruego, catalán, gallego, croata, esloveno, asturiano y galés.
# - Servicio GIL: http://www.corpus.unam.mx/servicio-freeling/
# 
# Entre sus principales servicios ofrecidos por la biblioteca FreeLing, se pueden citar:
# <ul>
#     <li>Tokenización de texto.</li>
#     <li>División de oraciones.</li>
#     <li>Análisis morfológico.</li>
#     <li>Tratamiento de sufijos, retokenización de pronombres clíticos.</li>
#     <li>Reconocimiento de palabras compuestas.</li>
#     <li>Reconocimiento flexible de múltiples palabras.</li>
#     <li>Separación por contracción.</li>
#     <li>Predicción probabilística de categorías de palabras desconocidas.</li>
#     <li>Codificación fonética.</li>
#     <li>Búsqueda basada en SED para palabras similares en el diccionario</li>
#     <li>Detección y clasificación de entidad nombrada.</li>
#     <li>Reconocimiento de fechas, números, relaciones, moneda y magnitudes físicas (velocidad, peso, temperatura, densidad, etc.)</li>
#     <li>Etiquetado de POS.</li>
#     <li>Análisis de poca profundidad basado en gráficos.</li>
#     <li>Anotación sensorial basada en WordNet y desambiguación.</li>
#     <li>Análisis de dependencia basado en reglas.</li>
#     <li>Análisis de dependencia estadística.</li>
#     <li>Etiquetado semántico estadístico de roles.</li>
#     <li>Resolución de la referencia.</li>
#     <li>Extracción semántica de grafos.</li>
# </ul>

# In[19]:


import requests

# Archivo a ser enviado
files = {'file': open('ElRamoAzul.txt', 'rb')}

# Parámetros
params = {'outf': 'tagged', 'format': 'json'}
# params = {'outf': 'dep', 'format': 'json'}

# Enviar petición
url = "http://www.corpus.unam.mx/servicio-freeling/analyze.php"
r = requests.post(url, files=files, params=params)
try:
    # Convertir de formato json
    obj = r.json()

    # Ejemplo, obtener todos los lemas
    for sentence in obj:
        for word in sentence:
            # print (word)
            print (word['lemma'],word['token'],word['tag'])
except:
    import traceback
    print( traceback.format_exc() )


# ## Stanford CoreNLP – software PLN
# - Descarga y Documentación: https://stanfordnlp.github.io/CoreNLP/
# - Lenguajes humanos soportados: La distribución básica proporciona archivos modelo para el análisis de inglés bien editado, pero el motor es compatible con modelos para otros idiomas, pero tiene modelos empaquetados para árabe, chino, francés, alemán y español.
# 
# ## Treetagger
# - Documentación: http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/
# - Lenguajes: alemán, inglés, francés, italiano, danés, holandés, español, búlgaro, ruso, portugués, gallego, griego, chino, swahili, eslovaco, esloveno, latín, estonio, polaco, rumano, checo, Coptos y antiguos textos en francés y es adaptable a otros idiomas si hay disponible un léxicon y un corpus de entrenamiento etiquetado manualmente.
# 
# ## CitiusTagger and CitiusNec
# - Documentación: https://gramatica.usc.es/pln/tools/CitiusTools.html
# - Una herramienta de clasificación de etiquetas POS y Entidades nombradas para portugués, inglés, gallego y español
# - Etiquetas: Eagles

# ## Analizando Moby Dick

# In[94]:


with open('moby.txt', 'r') as f:
    moby_raw = f.read()


# ### Ejemplo 1
# 
# ¿Cuántos tokens (palabras y símbolos de puntuación) hay en texto1?

# In[3]:


def ejemplo1():
    
    ''''''

ejemplo1()


# ### Ejemplo 2
# 
# ¿Cuántos tokens únicos (palabras únicas y símbolos de puntuación) tiene texto1?

# In[4]:


def ejemplo2():
    
    '''def ejemplo2():
    tokens = nltk.word_tokenize(moby_raw)
    print( len(set(tokens) ) )'''

ejemplo2()


# ### Ejemplo 3
# 
# ¿Después de lematizar los verbos, cuántos tokens únicos tiene texto1?

# In[5]:


from nltk.stem import WordNetLemmatizer

def example_three():

    '''from nltk.stem import WordNetLemmatizer

def example_three():
    tokens = nltk.word_tokenize(moby_raw)

    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(t) for t in tokens]
    print(len(set(tokens)))

example_three()'''

example_three()


# ### Pregunta 1
# 
# ¿Cuál es la diversidad léxica del texto de entrada? (ratio entre tokens únicos y el número total de tokens)

# In[6]:


def respuesta1():
    ''''''

respuesta1()


# ### Pregunta 2
# 
# ¿Qué porcentaje de tokens es 'whale'or 'Whale'?

# In[7]:


import re
def respuesta2():
    '''import re
def respuesta2():
    pattern = re.compile(r'\b[w|W]hale\b', )
    print(len(pattern.findall(moby_raw)) / 254989)

respuesta2()'''

respuesta2()


# ### Pregunta 3
# ¿Cuáles son los 20 tokens más frecuentes (únicos) en el texto? ¿Cuál es su frecuencia?
# 
# *Esta función debería devolver una lista de 20 tuplas donde cada tupla tiene la forma `(token, frecuencia)`. La lista debe ordenarse en orden descendente de frecuencia.*

# In[ ]:


tag_fd = nltk.FreqDist(tag for tag in tokens_dist)


# In[8]:


from collections import Counter
def respuesta3():
    
    ''''''
    
respuesta3()


# ### Pregunta 4
# 
# ¿Qué tokens tienen una longitud superior a 5 y frecuencia de más de 150?
# 
# *Esta función debería devolver una lista ordenada alfabeticamente de los tokens que coinciden con las restricciones anteriores. Para ordenar la lista, usar `sorted ()`*

# In[ ]:





# In[9]:


def respuesta4():
    
    ''''''

respuesta4()


# ### Pregunta 5
# 
# Encontrar la palabra mas larga en texto1 y la longitud de dicha palabra.
# 
# *Esta función debe devolver una tupla `(palabra, longitud)`.*

# In[104]:


def respuesta5():
    
    ''''''

respuesta5()


# ### Pregunta 6
# 
# Qué palabras únicas tienen una frecuencia de más de 2000? ¿Cuál es su frecuencia?
# 
# "Sugerencia: es conveniente utilizar` isalpha () `para comprobar si el token es una palabra y no una puntuación".
# 
# * Esta función debe devolver una lista de tuplas de la forma `(frecuencia, palabra)` ordenadas en orden descendente de frecuencia. *

# In[10]:


def respuesta6():
    
    ''''''

respuesta6()


# ### Pregunta 7
# ¿Cuál es el número promedio de tokens por oración?

# In[11]:


import statistics
def respuesta7():
    
    ''''''

respuesta7()


# ### Pregunta 8
# 
# ¿Cuáles son las 5 etiquetas gramaticales más frecuentes en el texto? ¿Cuál es su frecuencia?
# 
# * Esta función debe devolver una lista de tuplas de la forma `(etiqueta_gramatical, frecuencia)` ordenadas en orden descendente de frecuencia. *

# In[12]:


def respuesta8():
    
    ''''''
    
respuesta8()


# In[ ]:





# ## Tarea

# 1. A partir del texto “El Ramo Azul”, de Octavio Paz, publicado en el libro español “Arenas movedizas” en 1949.
# 
# <p>“El ramo azul” trata de un viajero que pasa una noche en un pueblo inquietante. La trama se centra en el diálogo que ocurre cuando un hombre débil se acerca al narrador para intentar sacarle los ojos. Lo que expone Paz, con los ojos como un símbolo, es que hay límites para la percepción.</p>
# 
# a) ¿Cuántas palabras hay en el texto?
# 
# b) ¿Cuántas palabras diferentes existen?
# 
# c) ¿Qué cantidad de sustantivos, adjetivos y verbos posee el texto?
# 
# •	Para el inciso C) puede tomarse en cuenta la ayuda de otras herramientas implementadas. Ejemplo: el servicio FreeLing: http://www.corpus.unam.mx/servicio-freeling/ 
# 

# In[ ]:





# In[ ]:




