import pandas as pd
import numpy as np
import re
from string import punctuation
from time import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from itertools import chain
from collections import Counter
from gensim.models import word2vec
import pickle


def calc_len(df, q1, q2):
    df['n_words1'] = df.apply(axis=1, func= lambda x:len(x[q1].split(' ')) )
    df['n_words2'] = df.apply(axis=1, func= lambda x:len(x[q2].split(' ')) )
    
    df['len_q1'] = df.apply(axis=1, func=lambda x: len(x[q1]))
    df['len_q2'] = df.apply(axis=1, func=lambda x: len(x[q2]))

    df['avg_len_words1'] = df['n_words1']/df['len_q1']
    df['avg_len_words2'] = df['n_words2']/df['len_q2']
    df.set_value(df[df.avg_len_words2 == np.inf].index, col = 'avg_len_words2',value=1000)
    df.set_value(df[df.avg_len_words1 == np.inf].index, col = 'avg_len_words1', value=1000)

    df['diff_avg_len_words'] = df['avg_len_words1'] - df['avg_len_words2']


def add_length_letter(df):
    df['diff_length_q1q2'] = df.apply(
        axis=1, func=lambda x: abs(x['n_words1'] - x['n_words2']))


def add_length_words(df):
    df['diff_length_nw1nw2'] = df.apply(
        axis=1, func=lambda x: abs(x['len_q1'] - x['len_q2']))


def add_simple_matching(df, question1, question2, n_word=0):
    df[str(n_word + 1) + '_word_match'] = df.apply(
        axis=1,
        func=
        lambda x: False if ((x['n_words1'] < n_word + 1) or (x['n_words2'] < n_word + 1)) 
        else x[question1].split(' ')[n_word] == x[question2].split(' ')[n_word]
    )

	
def add_double_matching(df, n_w1, n_w2):
    df[str(n_w1 + 1) + '_' + str(n_w2 + 1) + '_word_match'] = df.apply(
        axis=1,
        func=
        lambda x: (x[str(n_w1 + 1) + '_word_match'] & x[str(n_w2 + 1) + '_word_match'])
    )

	
def add_triple_matching(df, n_w1, n_w2, n_w3):
    df[str(n_w1+1)+'_'+str(n_w2+1)+'_'+str(n_w3+1)+'_word_match'] = df.apply(axis=1,
                                                             func=lambda x:
                                                             (x[str(n_w1+1)+'_word_match']
                                                              & x[str(n_w2+1)+'_word_match']
                                                              & x[str(n_w3+1)+'_word_match']) )
	
	
def create_add_vars(train_df, q1, q2, n_first_word=3):
    calc_len(train_df, q1, q2)
    add_length_letter(train_df)
    add_length_words(train_df)
    add_simple_matching(train_df, q1, q2, -1)

    for i in range(0,n_first_word):
        add_simple_matching(train_df, q1, q2,  i)
    
    for i in range(0,n_first_word):
        for j in range(i+1,n_first_word):
            add_double_matching(train_df, i, j)
            print("combo", i, j, "done")
    
    for i in range(0,n_first_word):
        for j in range(i+1,n_first_word):
            for k in range(j+1,n_first_word):
                add_triple_matching(train_df, i, j, k)
                print("combo", i, j, k, "done")
    remove_nondiff_vars = ['n_words1','n_words2','len_q1','len_q2','avg_len_words1','avg_len_words2']
    train_df.drop(remove_nondiff_vars, axis=1, inplace=True)
        
		
def get_first_weight(x):
    y = x.split()[0].lower()
    if y in qcat:
        return qcat[y]
    else:
        return 1
    
	
def add_first_unique(train_df):
    train_df['first_unique1'] = train_df.apply(axis=1, func=lambda x:get_first_weight(x['question1_final']))
    train_df['first_unique2'] = train_df.apply(axis=1, func=lambda x:get_first_weight(x['question2_final']))
    train_df['sum_first_log'] = np.log(train_df['first_unique1'] + train_df['first_unique2'])
    train_df.drop(['first_unique1', 'first_unique2'], inplace=True, axis=1)

	
def match_capital_words(a, b):
    a = {x for x in a.split()[1:] if x[0].isupper()}
    b = {x for x in b.split()[1:] if x[0].isupper()}
    
    # insersection / the whole words
    tot = set(list(a) + list(b))
    if len(tot) == 0:
        return 0
    return len(tot - (b - a) - (a - b))/len(tot)
	
	
def calc_match_capital(train_df, question1, question2):
    train_df['cap_words_match'] = train_df.apply(
        axis=1,
        func=lambda x: f.match_capital_words(x[question1], x[question2]))
    train_df['word_share'] = train_df.apply(
        lambda x: normalized_word_share(x[question1], x[question2]), axis=1)
    train_df['word_share_tfidf'] = train_df.apply(
        lambda x: tfidf_word_match_share(x[question1], x[question2]), axis=1)


def tfidf_word_match_share(q1, q2):
    q1words = {}
    q2words = {}
    for word in str(q1).lower().split():
        q1words[word] = 1
    for word in str(q2).lower().split():
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [
        get_weight_words(w) for w in q1words.keys() if w in q2words
    ] + [get_weight_words(w) for w in q2words.keys() if w in q1words]
    total_weights = [get_weight_words(w) for w in q1words
                     ] + [get_weight_words(w) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

	
def normalized_word_share(q1, q2):
    q1words = {}
    q2words = {}
    for word in str(q1).lower().split():
        q1words[word] = 1
    for word in str(q2).lower().split():
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (
        len(q1words) + len(q2words))
    return R

	
# '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' ## value in string.puntuation
def create_punctuation_features(prefix):
    punc_vars = {}
    punc_vars[prefix+'n_questmark'] = lambda text: text.count('?') # quante volte c'è: ?
    punc_vars[prefix+'end_questmark'] = lambda text: '?' in text[-1] # conclude con ?
    punc_vars[prefix+'end_point'] = lambda text: '.' in text[-1] # conclude con .
    punc_vars[prefix+'n_points'] = lambda text: text.count('.')
    punc_vars[prefix+'dollar'] = lambda text: '$' in text # contiene $
    punc_vars[prefix+'n_2virgolette'] = lambda text: text.count('\"')//2
    punc_vars[prefix+'n_virgoletta'] = lambda text: text.count("\'")
    punc_vars[prefix+'at_in_text'] = lambda text: '@' in text
    punc_vars[prefix+'n_coma'] = lambda text: text.count(',')
    punc_vars[prefix+'quadra_in'] = lambda text: ('[' in text) and (']' in text) #non conta quante però
    punc_vars[prefix+'graffa_in'] = lambda text: ('{' in text) and ('}' in text)
    punc_vars[prefix+'tonda_in'] = lambda text: ('(' in text) and (')' in text)
    punc_vars[prefix+'and_in'] = lambda text: '&' in text
    punc_vars[prefix+'percentage_in'] = lambda text: '%' in text
    punc_vars[prefix+'dublepoint_in'] = lambda text: ':' in text
    punc_vars[prefix+'n_semicolumn'] = lambda text: text.count(';')
    punc_vars[prefix+'tag_in'] = lambda text: '#' in text
    punc_vars[prefix+'escamation_in'] = lambda text: '!' in text
    punc_vars[prefix+'hat_in'] = lambda text: '^' in text
    punc_vars[prefix+'or_in'] = lambda text: '|' in text
    def min_point_position(text):
        try:
            return min(np.array( re.search('\.', text).span() )/len(text)) 
        except AttributeError: 
            return 0.
    punc_vars[prefix+'min_point_position'] = min_point_position
        
    def min_semicolumn_position(text):        
        try:
            return min(np.array( re.search('\;', text).span() )/len(text))
        except AttributeError: 
            return 0.
    punc_vars[prefix+'min_semicolumn_position'] = min_semicolumn_position
            
    def min_comma_position(text):    
        try: 
            return min(np.array( re.search('\,', text).span() )/len(text))
        except AttributeError: 
            return 0.
    punc_vars[prefix+'min_comma_position'] = min_comma_position
        
    punc_vars[prefix+'n_virgoletta2'] = lambda text: text.count('`')
    punc_vars[prefix+'tilde_in'] = lambda text: '~' in text
    punc_vars[prefix+'under_in'] = lambda text: '_' in text
    punc_vars[prefix+'slash1_in'] = lambda text: text.count('\'')
    punc_vars[prefix+'slash2_in'] = lambda text: text.count('/')        
    punc_vars[prefix+'star_in'] = lambda text: '*' in text
    return punc_vars

	
def add_punctuations(train_df, question1, question2):
    for q in (question1, question2):
        train_df[q] = train_df.apply(axis=1, func=lambda x: x[q] if len(x[q]) != 0 else " ")
    for q in (question1, question2):
        for k,v  in create_punctuation_features(q).items():
            train_df[k] = train_df[q].map(v)

diff_punctuation = [
    'n_questmark', 'end_questmark', 'end_point', 'n_points', 'dollar',
    'n_2virgolette', 'n_virgoletta', 'at_in_text', 'n_coma', 'quadra_in',
    'graffa_in', 'tonda_in', 'and_in', 'percentage_in', 'dublepoint_in',
    'n_semicolumn', 'tag_in', 'escamation_in', 'hat_in', 'or_in',
    'min_point_position', 'min_semicolumn_position', 'min_comma_position',
    'n_virgoletta2', 'tilde_in', 'under_in', 'slash1_in', 'slash2_in',
    'star_in'
]


def create_diff_punteggiatura(df, question1, question2):
    add_punctuations(df, question1, question2)
    for v in diff_punctuation:
        if df['question1'+v].dtype == 'bool':
            df['diff_or_'+v] = df[question1+v] | df[question2+v]
            df['diff_and_'+v] = df[question1+v] & df[question2+v]
        else:
            df['diff_'+v] = np.abs(df[question1+v] - df[question2+v])
    df.drop(['question1' + v for v in diff_punctuation], axis=1, inplace=True)
    df.drop(['question2' + v for v in diff_punctuation], axis=1, inplace=True)        

sub_replace = {
    r"\'ve": " have ",
    r"n't": " not ",
    r"I'm": "I am",
    r" m ": " am ",
    r"\'re": " are ",
    r"\'ll": " will ",
    r"he\'s": "he is",
    r"she\'s": "she is",
    "I'm": "I am",
    "i'm": "i am",
    "What's": "What is",
    "what's": "what is",
    "Who's": "Who is",
    "How's": "How is",
    "Where's": "Where is",
    "Why's": "Why is",
    "It's": "It is",
    "There's": "There is",
    "There're": "There are",
    "Let's": "Lets"
}


def clear_text_first_step(text):
    text = re.sub("HOw", "How", text)
    text = re.sub("WHAT", "What", text)
    text = re.sub("HOW", "How", text)
    text = re.sub("IS", "is", text)

    for k in sub_replace:
        text = re.sub(k, sub_replace[k], text)
        text = re.sub(k.replace("'", '’'), sub_replace[k], text)

    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r" mail", "email", text)

    text = re.sub("U.S.", "USA", text)
    text = re.sub("US", "USA", text)
    text = re.sub(r" usa ", " USA ", text)
    text = re.sub(r" u s ", " USA ", text)
    text = re.sub(r" USAA ", " USA ", text)
    text = re.sub(r" uk ", " UK ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"banglore", "Banglore", text)
 
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)

    text = re.sub(r"\'s", " ", text)  # genitivo sassone rimosso
    text = re.sub(r"’s", " ", text)  # genitivo sassone rimosso

    text.replace(' + ', 'plus')
    text.replace(' - ', 'minus')
    text.replace(' > ', 'major')
    text.replace(' < ', 'minor')
    text.replace(' = ', 'equal')

    text.replace('-', ' - ')
    text = re.sub('\s+', ' ', text).strip(
    )  # remove more space, tabs, whitespace-like and trail space    
    return text

	
def remove_virgolette(s):
    s = s + ' '
    if '?"' in s: s = s.replace('?"', '" ?')
    if "?'" in s: s = s.replace("?'", "' ?")
    for sub in re.finditer("""\s['"“][\-\w\s]+['"“][\s.,?;:!]""",
                           s):
        s = re.sub(
            sub.group(0),
            sub.group(0)[0] + sub.group(0)[2:-2] + sub.group(0)[-1], s)
    return s

	
def remove_parentesi(s):
    s = s + ' '
    for sub in re.finditer("\s[{][\-\w\s]+[}][\s.,?;:!]",
                           s):
        s = re.sub(
            sub.group(0),
            sub.group(0)[0] + sub.group(0)[2:-2] + sub.group(0)[-1], s)
        
    s = s.replace('(',' ')
    s = s.replace(')',' ')
    s = s.replace('[',' ')
    s = s.replace(']',' ')
    return s    

	
def clear_correzioni_mano(text):
    text = re.sub(" iphones ", " iPhone ", text)
    text = re.sub(" ireland ", " Ireland' ", text)
    text = re.sub(" mustn ", " must ", text)
    text = re.sub(" linux ", " Linux ", text)
    text = re.sub(" wouldn ", " would not ", text)
    text = re.sub(" videoes ", " videos ", text)
    text = re.sub(" tv ", " TV ", text)
    text = re.sub(" google ", " Google ", text)
    text = re.sub(" memorise ", " memorize ", text)
    text = re.sub(" faught ", " fought ", text)
    text = re.sub(" cena ", " Cena ", text)
    text = re.sub(" lesner ", " Lesner ", text)
    text = re.sub(" hollywood ", " Hollywood ", text)
    text = re.sub(" anmount ", " amount ", text)
    text = re.sub(" hoywood ", " Hollywood ", text)
    text = re.sub(" cantonese ", " Cantonese ", text)
    text = re.sub(" otherthan ", " other than ", text)
    text = re.sub(" mumbai ", " Mumbai ", text)
    text = re.sub(" wikipedia ", " Wikipedia ", text)
    text = re.sub(" textfields ", " text fields ", text)
    text = re.sub(" ajax ", " Ajax ", text)
    text = re.sub(" pls ", " please ", text)
    text = re.sub(" couldn ", " could not ", text)
    text = re.sub(" calcutta ", " Calcutta ", text)
    text = re.sub(" doesnt ", " does not ", text)
    text = re.sub(" fIght ", " fight ", text)
    text = re.sub(" txt ", "> text ", text)
    text = re.sub(" whther ", " whethere ", text)
    text = re.sub(" feelns ", " feelings ", text)
    text = re.sub(" sudd ", " suddenly ", text)
    text = re.sub(" stl ", " steal ", text)
    text = re.sub(" india ", " India ", text)
    text = re.sub(" Plz ", " please ", text)
    text = re.sub(" engg ", " Eng ", text)
    text = re.sub(" eng ", " Eng ", text)
    text = re.sub(" olympians ", " Olympians ", text)
    text = re.sub(" offence ", " offense ", text)
    text = re.sub(" bulgarians ", " Bulgarians ", text)
    text = re.sub(" siemens ", " Siemens ", text)
    text = re.sub(" wasn ", " was not ", text)
    text = re.sub(" clinton's ", " Clinton ", text)
    text = re.sub(" portland ", " Portland ", text)
    text = re.sub(" recognise ", " recognize ", text)
    text = re.sub(" adams ", " Adamns ", text)
    text = re.sub(" didnt ", " did not ", text)
    text = re.sub(" taylor ", " Taylor ", text)
    text = re.sub(" youtube ", " YoutTube ", text)
    text = re.sub(" goverment ", " government ", text)
    text = re.sub(" korean ", " Korean ", text)
    text = re.sub(" paypal ", " PayPal ", text)
    text = re.sub(" isn ", " is not ", text)
    text = re.sub(" facebook ", " Facebook ", text)
    text = re.sub(" mhz ", " MHz ", text)
    text = re.sub(" samsung ", " Samsung ", text)
    text = re.sub(" womans ", " woman ", text)
    text = re.sub(" german ", " German ", text)
    text = re.sub(" america ", " America ", text)
    text = re.sub(" mosquitos ", " mosquitoes ", text)
    text = re.sub(" melbourne ", " Melbourne ", text)
    text = re.sub(" dj ", " DJ ", text)
    text = re.sub(" behaviour ", " behavior ", text)
    text = re.sub(" hasn ", " has not ", text)
    text = re.sub(" phd ", " PhD ", text)
    text = re.sub(" aren ", " are not ", text)
    text = re.sub(" ethernet ", " Ethernet ", text)
    text = re.sub(" uk ", " UK ", text)
    text = re.sub(" realise ", " realize ", text)
    text = re.sub(" brisbane ", " Brisbane ", text)
    text = re.sub(" organisation ", " organization ", text)
    text = re.sub(" aftr ", " after ", text)
    text = re.sub(" russian ", " Russian ", text)
    text = re.sub(" nonpolar ", " non polar ", text)
    text = re.sub(" pc ", " PC ", text)
    text = re.sub(" othet ", " other ", text)
    text = re.sub(" nokia ", " Nokia ", text)
    text = re.sub(" boolean ", " Boolean ", text)
    text = re.sub(" analyse ", " analyze ", text)
    text = re.sub(" centres ", " centers ", text)
    text = re.sub(" ramadan ", " Ramadan ", text)
    text = re.sub(" latin ", " Latin ", text)
    text = re.sub(" weren ", " were not ", text)
    text = re.sub(" immedietly ", " immediately ", text)
    text = re.sub(" bollywood ", " Bollywood ", text)
    text = re.sub(" conentration ", " concentration ", text)
    text = re.sub(" benifit ", " benefit ", text)
    text = re.sub(" oppurtunities ", " opportunities ", text)
    text = re.sub(" filipino ", " Filipino ", text)
    text = re.sub(" netflix ", " Netflix ", text)
    text = re.sub(" indians ", " Indians ", text)
    text = re.sub(" opensource ", " open source ", text)
    text = re.sub(" atlanta ", " Atlanta ", text)
    text = re.sub(" microsoft ", " Microsoft ", text)
    text = re.sub(" colour ", " color ", text)
    text = re.sub(" cse ", " CSE ", text)
    text = re.sub(" jane ", " Jane ", text)
    text = re.sub(" exsts ", " exist ", text)
    text = re.sub(" persob ", " person ", text)
    text = re.sub(" centre ", " center ", text)
    text = re.sub(" radeon ", " Radeon ", text)
    text = re.sub(" postgraduation ", " post graduation ", text)
    text = re.sub(" suez ", " Suez ", text)
    text = re.sub(" illuminati ", " Illuminati ", text)
    text = re.sub(" analytics ", " analytic ", text)
    text = re.sub(" italian ", " Italian ", text)
    text = re.sub(" excercises ", " exercises ", text)
    text = re.sub(" favour ", " favor ", text)
    text = re.sub(" smartphones ", " smartphone ", text)
    text = re.sub(" shouldn ", " should not ", text)
    text = re.sub(" didnot ", " did not ", text)
    text = re.sub(" friday ", " Friday ", text)
    text = re.sub(" monday ", " Monday ", text)
    text = re.sub(" americans ", " American ", text)
    text = re.sub(" hasn ", " has not ", text)
    text = re.sub(" michael ", " Michael ", text)
    text = re.sub(" verizon ", " Verizon ", text)
    text = re.sub(" hitler ", " Hitler ", text)
    text = re.sub(" fermi ", " Fermi ", text)
    text = re.sub(" whatsapp ", " Whatsapp ", text)
    text = re.sub(" messagess ", " messages ", text)
    text = re.sub(" africa ", " Africa ", text)
    text = re.sub(" weakneses ", " weakness ", text)
    text = re.sub(" nikon ", " Nikon ", text)
    text = re.sub(" capricorn ", " Capricorn ", text)
    text = re.sub(" romania ", " Romania ", text)
    text = re.sub(" favourite ", " favorite ", text)
    text = re.sub(" startups ", " startup ", text)
    text = re.sub(" spanish ", " Spanish ", text)
    text = re.sub(" preparegravitation ", " prepare gravitation ", text)
    text = re.sub(" compulsary ", " compulsory ", text)
    text = re.sub(" workin ", " working ", text)
    text = re.sub(" syria ", " Syria ", text)
    text = re.sub(" immigants ", " immigrants ", text)
    text = re.sub(" benedict ", " Benedict ", text)
    text = re.sub(" legssss ", " legs ", text)
    text = re.sub(" france ", " France ", text)
    text = re.sub(" watsup ", " Whatsapp ", text)
    text = re.sub(" arya ", " Arya ", text)
    text = re.sub(" handjob ", " Handjob ", text)
    text = re.sub(" europe ", " Europe ", text)
    text = re.sub(" shoud ", " should ", text)
    text = re.sub(" paypal ", " Paypal ", text)
    text = re.sub(" upto ", " up to ", text)
    text = re.sub(" paris  ", " Paris ", text)
    text = re.sub(" sql ", " SQL ", text)
    text = re.sub(" hitman ", " Hitman ", text)
    text = re.sub(" lagrangian ", " Lagrangian ", text)
    text = re.sub(" dvd ", " DVD ", text)
    text = re.sub(" donald ", " Donald ", text)
    text = re.sub(" enigneering ", " engineering ", text)
    text = re.sub(" mightn ", " might ", text)
    text = re.sub(" defence ", " defense ", text)
    text = re.sub(" iranian ", " Iranian ", text)
    text = re.sub(" increse ", " increase ", text)
    text = re.sub(" india ", " India ", text)
    text = re.sub(" hairloss ", " hair loss ", text)
    text = re.sub(" volumetry ", " volume try ", text)
    text = re.sub(" americans ", " Americans ", text)
    text = re.sub(" quora ", " Quora ", text)
    text = re.sub(" eligiblty ", " eligibility ", text)
    text = re.sub(" english ", " English ", text)
    text = re.sub(" indian ", " Indian ", text)
    text = re.sub(" bangalore ", " Bangalore ", text)
    text = re.sub(" emoji ", " emoji ", text)
    text = re.sub(" ielts ", " IELTS ", text)
    text = re.sub(" ahmedabad ", " Ahmadabad ", text)
    text = re.sub(" frac ", " Frac ", text)
    text = re.sub(" sociall ", " socially ", text)
    text = re.sub(" philippines ", " Philippines ", text)
    text = re.sub(" java ", " Java ", text)
    text = re.sub(" intraday ", " Intraday ", text)
    text = re.sub(" mightn ", " might ", text)
    text = re.sub(" delhi ", " Delhi ", text)
    text = re.sub(" saturn ", " Saturn ", text)
    text = re.sub(" youtube ", " Youtube ", text)
    text = re.sub(" noida ", " Noida ", text)
    text = re.sub(" lynda ", " Lynda ", text)
    text = re.sub(" demonetisation ", " demonetization ", text)
    text = re.sub(" html ", " HTML ", text)
    text = re.sub(" dissprove ", " disprove ", text)
    text = re.sub(" nlp ", " NLP ", text)
    text = re.sub("\(nlp\)", "\(NLP\)", text)
    text = re.sub(" rollerblade ", " Rollerblade ", text)
    text = re.sub(" vlc ", " VLC ", text)
    text = re.sub(" rolex ", " Rolex ", text)
    text = re.sub(" november ", " November ", text)
    text = re.sub(" indians ", " Indians ", text)
    text = re.sub("inflammatories ", "inflammatory ", text)
    text = re.sub(" absorps ", " absorbs ", text)
    text = re.sub(" kat.cr ", " Kat.cr ", text)
    text = re.sub(" ibm ", " IBM ", text)
    text = re.sub(" centra\)", " central\)", text)
    text = re.sub(" centra ", " central ", text)
    text = re.sub(" uk ", " UK ", text)
    text = re.sub(" pdf ", " PDF ", text)
    text = re.sub(" ebook ", " Ebook ", text)
    text = re.sub(" sydney ", " Sydney ", text)
    text = re.sub(" samsung ", " Samsung ", text)
    text = re.sub(" usa ", " USA ", text)
    text = re.sub(" traveller ", " traveler ", text)
    text = re.sub(" jaipur ", " Jaipur ", text)
    text = re.sub(" pablo ", " Pablo ", text)
    text = re.sub(" ebay ", " eBay ", text)
    text = re.sub(" Ebay ", " eBay ", text)
    text = re.sub(" EBAY ", " eBay ", text)
    text = re.sub(" whatsapp ", " Whatsapp ", text)
    text = re.sub(" imessage ", " Imessage ", text)
    text = re.sub(" slary ", " salary ", text)
    text = re.sub(" isis ", " ISIS ", text)
    text = re.sub(" blow job", " blowjob", text)
    text = re.sub(" eu ", " EU ", text)
    text = re.sub(" favourite ", " favorite ", text)
    text = re.sub(" reactjs ", " react ", text)
    text = re.sub(" pakistan ", " Pakistan ", text)
    text = re.sub(" stanford ", " Stanford ", text)
    text = re.sub(" harvard ", " Harvard ", text)
    text = re.sub(" wharton ", " Wharton ", text)
    text = re.sub(" saturn ", " Saturn ", text)
    text = re.sub(" existance ", " existence ", text)
    text = re.sub(" gb ", " GB ", text)
    text = re.sub(" poeple ", " people ", text)
    text = re.sub(" forex ", " Forex ", text)
    text = re.sub(" katrina ", " Katrina ", text)
    text = re.sub(" decison ", " decision ", text)
    text = re.sub(" snapchat ", " Snapchat ", text)
    text = re.sub(" rollerblade ", " Rollerblade ", text)
    text = re.sub(" mba ", " MBA ", text)
    text = re.sub(" anime ", " Anime ", text)
    text = re.sub(" disney ", " Disney ", text)
    text = re.sub(" schengen ", " Schengen ", text)
    text = re.sub(" singapore ", " Singapore ", text)
    text = re.sub(" ramayan ", " Ramayan ", text)
    text = re.sub(" gmail ", " Gmail ", text)
    text = re.sub(" madheshi ", " Madheshi ", text)
    text = re.sub(" germany ", " Germany ", text)
    text = re.sub(" instagram ", " Instagram ", text)
    text = re.sub(" connecticut ", " Connecticut ", text)
    text = re.sub(" php ", " PHP ", text)
    text = re.sub(" reaso? ", " reason?", text)
    text = re.sub(" japanese ", " Japanese ", text)
    text = re.sub(" gf ", " girlfriend ", text)
    text = re.sub(" mumbai ", " Mumbai ", text)
    text = re.sub(" robert ", " Robert ", text)
    text = re.sub(" linkedin ", " Linkedin ", text)
    text = re.sub(" maharashtrian ", " Maharashtrian ", text)
    text = re.sub(" bollywood ", " Bollywood ", text)
    text = re.sub(" enginnering ", " engineering ", text)
    text = re.sub(" rattata ", " Rattata ", text)
    text = re.sub(" magikarp ", " Magikarp ", text)
    text = re.sub(" islam ", " Islam ", text)
    text = re.sub(" atleast ", " at least ", text)
    text = re.sub(" atleast?", " at least?", text)
    text = re.sub(" schengen ", " Schengen ", text)
    text = re.sub(" engeneering ", " engineering ", text)
    text = re.sub(" casanova ", " Casanova ", text)
    text = re.sub(" feelngs ", " feelings ", text)
    text = re.sub(" photoshop ", " Photoshop ", text)
    text = re.sub(" canada ", " Canada ", text)
    text = re.sub(" holland ", " Holland ", text)
    text = re.sub(" hollywood ", " Hollywood ", text)
    text = re.sub(" chelsea ", " Chelsea ", text)
    text = re.sub(" modernizaton ", " modernization ", text)
    text = re.sub(" instagrammer ", " Instagrammer ", text)
    text = re.sub(" thailand ", " Thailand ", text)
    text = re.sub(" chinese ", " Chinese ", text)
    text = re.sub(" corrrect ", " correct ", text)
    text = re.sub(" hillary ", " Hillary ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" undergraduation ", " undergraduate ", text)
    text = re.sub(r" qoura ", " Quora ", text)
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" latop", " laptop", text)
    text = re.sub(r" permantley ", " permanently ", text)
    text = re.sub(r" connectionn ", " connection ", text)
    text = re.sub(r" sylabus ", " syllabus ", text)
    text = re.sub(r" insititute ", " institute ", text)
    text = re.sub(r" sequrity ", " security ", text)
    text = re.sub(r"(?=[a-zA-Z])ig ", "ing ", text)
    text = re.sub(r" addmision ", " admission ", text)
    text = re.sub(r" begineer ", " beginner ", text)
    text = re.sub(r" wtiter ", " writer ", text)
    text = re.sub(r" litrate ", " literate ", text)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" programmning ", " programming ", text)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" unseccessful ", " unsuccessful ", text)
    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    return text

	
def clear_first(text):
    text = remove_virgolette(text)
    text = remove_parentesi(text)
    text = clear_text_first_step(text)
    # text = clear_correzioni_mano(text)
    return text

my_stopwords = [
    'a', 'an', 'the', 'and', 'but', 'if', 'If', 'or', 'as', 'until', 'while', 'during', 'of',
    'at', 'by', 'for', 'with', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
	'over', 'under', 'again', 'ma', 'm', 'o', 's', 't', 'same', 'so',
    'too', 'very', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'this', 'that', 'these', 'those',
    'then','just','than','such','both','through','about',
]

def remove_stopwors(text, stopwords):
    text = text.split()
    text = [w for w in text if not w in stopwords]
    text = " ".join(text)
    return text

	
def remove_punctuations(text):
    text = ''.join([c if c not in punctuation else ' ' for c in text])
    text = re.sub('\s+', ' ', text).strip(
    )  # remove more space, tabs, whitespace-like and trail space    
    return text

	
sub_replace1 = {
    r"\'d": " d "
}


def clear_text_second_step(text):
    for k in sub_replace1:
        text = re.sub(k, sub_replace1[k], text)
        text = re.sub(k.replace("'", '’'), sub_replace1[k], text)
    return text

	
def clear_second(text):
    text = clear_correzioni_mano(text)
    text = clear_text_second_step(text)
    text = remove_punctuations(text)
    text = remove_stopwors(text, my_stopwords)
    return text

	
def add_clear_first(df, q):
    df[q+'_clear_1'] = df.apply(axis=1, func=lambda x: clear_first(x[q]))

	
def add_clear_second(df, q):
    df[q+'_clear_2'] = df.apply(axis=1, func=lambda x: clear_second(x[q]))

	
def countword(string,lista):
    return sum(lista[v] for v in string.split())
	
	
def get_bit_single_word(string, w):
    app = re.search(w, string)
    return app != None
	
	
####### TAGGER #####


from functools import reduce
from joblib import Parallel, delayed
import multiprocessing
from treetagger import TreeTagger
tt = TreeTagger(language='english')


def tag_phrase(text):
    return tt.tag(text)

	
def sum_quests_and_tag(u1, i, q):
    i = u1.index
    tet = ' #### '.join(u1[q].values)
    results = tag_phrase(tet)
    
    t = tet.split(' #### ')
    df = pd.DataFrame(results)
    idx = df[df[0] == '####'].index
    l = [results[1: idx[0]]]
    l.extend( [results[idx[i] + 1:idx[i + 1]] for i in range(len(idx) - 1)] )
    l.append(results[idx[-1] + 1:])
       
    if (len(l) != len(t)):
        print(len(l), len(t), i)
        print('different lenght in position {}'.format(df.index),end='\n\n')
        return
    df = pd.DataFrame(np.array([l, t]).T, columns=['tagger', q])
    df.to_csv('csv_question_tag/questions_tag_' + str(i.min()) + '_' +
              str(i.max()) + '.csv')
    return


def create_tagger_csv(unique_questions, step=3000):
    start = time()
    for i in range(0, len(unique_questions),step):
        sum_quests_and_tag(unique_questions.loc[i:i+step], i, 'questions_clear_1')
    stop = time()
    print((stop - start)/60, 'minutes')

	
def fix_creations_csv_tagger(start_stop, division=20):
    start = time()
    for v in start_stop:
        stepp = (v[1]-v[0])//division
        for j in range(v[0], v[1], stepp):
            sum_quests_and_tag(unique_questions.loc[j:j+stepp], j, 'questions_clear_1')
    stop = time()
    print((stop - start)/60, 'minutes')

	
def get_phrase_from_tagger(tagg_list):
    if len(tagg_list[0]) != 3: return ' '
    l = [
        c[2] if (c[2] != '<unknown>') and (c[2] != '@card@') else c[0]
        for c in tagg_list
    ]
    return reduce(lambda x, y: x + ' ' + y, l)

	
def convert_tag_to_phrase(df, question_after_tag, tagger):
    df[question_after_tag] = df.apply(
        axis=1, func=lambda x: get_phrase_from_tagger(x[tagger]))


####### END TAGGER #####


def create_submission_xgboost(test_df, features, model):
    d_test = xgb.DMatrix(test_df[features])
    y_submission = model.predict(d_test)
    test_df['predict_proba'] = y_submission
    submission = pd.DataFrame({'test_id': test_df['test_id'], 'is_duplicate': y_submission})
    submission.to_csv("submission.csv", index=False)
    print('submission.csv created')
   
   
def create_unique_questions(train_df, test_df):
    words = list(
        set(
            list(set(train_df['question1'])) +
            list(set(train_df['question2'])) + list(set(test_df['question1']))
            + list(set(test_df['question2']))))
    unique_questions = pd.DataFrame(
        words, columns=['questions']).reset_index(drop=True)
    return unique_questions

	
#### SOME VARIABLES ####


def get_SPOILER_bit(string, app=0):
    app = len(re.findall('SPOILER', string))
    app = app + len(re.findall('SPOILERS', string))
    app = app + len(re.findall('spoiler', string))
    app = app + len(re.findall('spoilers', string))
    app = app + len(re.findall('Spoiler', string))
    app = app + len(re.findall('Spoilers', string))
    if app > 0:
        return (True)
    return (False)

	
def get_math_bit(string, app=0):
    app = len(re.findall('[0-9]x ', string))
    app += len(re.findall(' x[0-9]', string))
    app += len(re.findall(' [0-9]x', string))
    app += len(re.findall('x[0-9] ', string))
    app += len(re.findall('y=', string))
    app += len(re.findall('y =', string))
    app += len(re.findall('x=', string))
    app += len(re.findall('x =', string))
    app += len(re.findall('x^', string))
    app += len(re.findall(' sen ', string))
    app += len(re.findall(' sin ', string))
    app += len(re.findall(' cos ', string))
    app += len(re.findall(' log ', string))
    app += len(re.findall(' ln ', string))
    app += len(re.findall(' e^', string))
    app += len(re.findall('\[math\]', string))
    app += len(re.findall('frac\{', string))
    if app > 0:
        return (True)
    return (False)


def get_diff_spoiler_math(df, q1, q2):
    df['spoiler_or'] = df.apply(
        axis=1,
        func=lambda x: get_SPOILER_bit(x[q1]) or get_SPOILER_bit(x[q2]))
    df['spoiler_and'] = df.apply(
        axis=1,
        func=lambda x: get_SPOILER_bit(x[q1]) and get_SPOILER_bit(x[q2]))
    df['math_or'] = df.apply(axis=1, func=lambda x: get_math_bit(x[q1]) or get_math_bit(x[q2]) )
    df['math_or'] = df.apply(axis=1, func=lambda x: get_math_bit(x[q1]) and get_math_bit(x[q2]) )


def normalized_word_share(question1, question2, dizionario, tipo):
    question1 = [x for x in question1.split() if dizionario.get(x) == tipo]
    question2 = [x for x in question2.split() if dizionario.get(x) == tipo]
    w1 = set(map(lambda word: word.lower().strip(), question1))
    w2 = set(map(lambda word: word.lower().strip(), question2))
    if len(w1) + len(w2) == 0: return 0
    return 1.0 * len(w1 & w2) / len(w1.union(w2))


def calc_match_capital2(train_df, q1, q2, tipo, diz):
    train_df['word_share ' + tipo] = train_df.apply(axis=1,
        func=lambda x: normalized_word_share(x[q1], x[q2], diz, tipo))


def get_sequenceones_similarity(q1,
                                q2,
                                tag_list=['VERB', 'NOUN', 'JJ', 'WRB']):
    if min(len(q1), len(q2)) == 0:
        return 0
    q1 = q1.split()
    q2 = q2.split()
    
    min_max_len = max(len(q1), len(q2))

    zeros_correction = []
    diff1 = min_max_len - len(q1)
    if diff1 > 0:
        zeros_correction = [0] * diff1 * len(tag_list)
    app1 = [1 if dizionario_max.get(i) == j else 0 for i in q1
            for j in tag_list] + zeros_correction

    zeros_correction = []
    diff2 = min_max_len - len(q2)
    if diff2 > 0:
        zeros_correction = [0] * diff2 * len(tag_list)
    app2 = [1 if dizionario_max.get(i) == j else 0 for i in q2
            for j in tag_list] + zeros_correction

    app = Counter(np.array(app1) + np.array(app2))
    return app[2] / min(len(q1), len(q2))


def getback_function(function):
    return(pd.Series(list(chain.from_iterable(function))))
	
	
def save_obj(obj, name):
    with open('./' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
		
		
def get_sequenceones_similarity2(q1,
                                 q2,
                                 voc,
                                 tag_dict={
                                     'VERB': [1, 0, 0, 0],
                                     'NOUN': [0, 1, 0, 0],
                                     'JJ': [0, 0, 1, 0],
                                     'WRB': [0, 0, 0, 1]
                                 }):
    if min(len(q1), len(q2)) == 0:
        return 0
    q1 = q1.split()
    q2 = q2.split()
    lenq1 = len(q1)
    lenq2 = len(q2)
    min_len = min(lenq1, len(q2))
    qq1 = list(
        chain.from_iterable([
            tag_dict.get(voc.get(i, 'NONLOTAGGO'), [3, 0, 0, 0]) for i in q1
        ]))
    qq2 = list(
        chain.from_iterable([
            tag_dict.get(voc.get(i, 'NONLOTAGGO'), [3, 0, 0, 0]) for i in q2
        ]))

    lunghezza = min(lenq1, lenq2)
    q, p = (qq1, qq2) if lenq1 == lunghezza else (qq2, qq1)
    x_max = abs(lenq1 - lenq2)

    len_tag_bit = len(tag_dict)
    app = [
        Counter(
            np.array(p[len_tag_bit * x:lunghezza * len_tag_bit +
                       len_tag_bit * x]) + np.array(q))
        for x in range(0, x_max + 1)
    ]
    out1 = max([(i[2]+i[6]) / min_len for i in app])
    out2 = max([(i[2]) / (1 if q.count(1) == 0 else q.count(1)) for i in app])
    return [[out1, out2]]


def get_variables_from_lists(q1, q2, lista):

    q1 = q1.lower()
    q2 = q2.lower()
    list1 = [s for s in lista if re.search(s, q1) is not None]
    list2 = [s for s in lista if re.search(s, q2) is not None]

    shared_list = [s for s in list1 if s in list2]
    
    llist1 = len(list1)
    llist2 = len(list2)
    lshared = len(shared_list)

    try:
        out1 = lshared / min(llist1, llist2)
    except ZeroDivisionError:
        out1 = 0
    try:
        out2 = lshared / min(len(q1.split()), len(q2.split()))
    except ZeroDivisionError:
        out2 = 0

    return ([out1], [out2], [abs(llist1 - llist2)])


def build_corpus(data, q1, q2):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in [q1, q2]:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus


def get_unknown_variables(q1, q2, english_vocab):
    q1 = q1.split()
    q2 = q2.split()

    unknown1 = [w1 for w1 in q1 if w1 not in english_vocab]
    unknown2 = [w2 for w2 in q2 if w2 not in english_vocab]

    shared_unknown = [un1 for un1 in unknown1 if un1 in unknown2]
    lun1 = len(unknown1)
    lun2 = len(unknown2)
    lshared = len(shared_unknown)

    try:
        out1 = lshared / min(lun1, lun2)
    except ZeroDivisionError:
        out1 = 0

    try:
        out2 = lshared / min(len(q1), len(q2))
    except ZeroDivisionError:
        out2 = 0

    return ([out1], [out2], [abs(lun1 - lun2)])


#### XGBOOST ####
def plot_var_imp_xgboost(model,mode='gain',ntop=-1):
    """Plot the vars imp for xgboost model, where mode = ['weight','gain','cover']  
    'weight' - the number of times a feature is used to split the data across all trees.
    'gain' - the average gain of the feature when it is used in trees
    'cover' - the average coverage of the feature when it is used in trees
    """
    importance = model.get_score(importance_type=mode)
    importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)
    if ntop == -1: ntop = len(importance)
    my_ff = np.array([i[0] for i in importance[0:ntop]])
    imp = np.array([i[1] for i in importance[0:ntop]])
    indices = np.argsort(imp)
    pos = np.arange(len(my_ff[indices]))+.5
    plt.figure(figsize=(20, 0.75*len(my_ff[indices])))
    plt.barh(pos, imp[indices], align='center')
    plt.yticks(pos, my_ff[indices],size=25)
    plt.xlabel('rank')
    plt.title('Feature importances ('+mode+')',size=25)
    plt.grid(True)
    plt.show()
    return importance


def plot_ROC_PrecisionRecall(y_test, y_pred):
    """Plot ROC curve and Precision-Recall plot"""
    fpr_clf, tpr_clf, _ = roc_curve(y_test, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    f1 = np.array([2 * p * r / (p + r) for p, r in zip(precision, recall)])
    f1[np.isnan(f1)] = 0
    t_best_f1 = thresholds[np.argmax(f1)]
    roc_auc = auc(fpr_clf, tpr_clf)
    plt.figure(figsize=(25, 25))
    # plot_ROC
    plt.subplot(221)
    plt.plot(
        fpr_clf,
        tpr_clf,
        color='r',
        lw=2,
        label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plot_PrecisionRecall
    plt.subplot(222)
    plt.plot(
        recall, precision, color='r', lw=2, label='Precision-Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precison-Recall curve')
    plt.legend(loc="lower right")

    plt.show()
    return {"roc_auc": roc_auc, "t_best_f1": t_best_f1}


def plot_auc_test_train(y_train, y_test, y_test_pred, y_train_pred):
    roc_auc_test = plot_ROC_PrecisionRecall(y_test, y_test_pred)
    roc_auc_train = plot_ROC_PrecisionRecall(y_train, y_train_pred)
    return roc_auc_test, roc_auc_train


def train_xgboost(df, features, target, param, other_par):
    df = df.sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.33, random_state=786)
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    model = xgb.train(param, dtrain=xg_train, evals=watchlist, **other_par)
    pred = model.predict(xg_test)
    pred_train = model.predict(xg_train)
    roc_auc_test, roc_auc_train = plot_auc_test_train(y_train, y_test, pred,
                                                      pred_train)
    var_imp = plot_var_imp_xgboost(model, ntop=40, mode='gain')
    return [model, roc_auc_test, roc_auc_train, var_imp]
	
	
def train_xgboost_w(df, features, target, param, other_par):
    df = df.sample(frac=1)
    X_train, X_test, y_train, y_test = f.train_test_split(
        df[features], df[target], test_size=0.33, random_state=786)
    r = df[target].value_counts()[0]/df[target].value_counts()[1]
    r = 1./0.165
    w_train = y_train*r
    w_train[y_train == 0] = 1
    w_test = y_test*r
    w_test[y_test == 0] = 1  
    xg_train = qd.xgb.DMatrix(X_train, label=y_train, weight=w_train)
    xg_test = qd.xgb.DMatrix(X_test, label=y_test, weight=w_test)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    model = qd.xgb.train(param, dtrain=xg_train, evals=watchlist, **other_par)
    pred = model.predict(xg_test)
    pred_train = model.predict(xg_train)
    roc_auc_test, roc_auc_train = qd.plot_auc_test_train(y_train, y_test, pred,
                                                      pred_train)
    var_imp = qd.plot_var_imp_xgboost(model, ntop=40, mode='gain')
    return [model, roc_auc_test, roc_auc_train, var_imp]
	
	
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import operator
import xgboost as xgb


def xgb_evaluate_gen(
        xg_train,
        xg_test,
        watchlist,
        params={'objective': 'binary:logistic',
                'silent': 1,
                'nthread': 10},
        other_par={
            'eval_metric': 'auc',
            'eta': 0.01,
            'num_boost_round': 10,
            'early_stopping_rounds': 50,
            'verbose_eval': 10
        },
        sgn=1):
    """Create the function to be optimized (example for xgboost)"""

    def xgb_evaluate(min_child_weight, colsample_bytree, max_depth, subsample,
                     gamma, alpha):
        """ Return the function to be maximized by the Bayesian Optimization,
        where the inputs are the parameters to be optimized and the output the 
        evaluation_metric on test set"""
        params['min_child_weight'] = int(round(min_child_weight))
        params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['subsample'] = max(min(subsample, 1), 0)
        params['gamma'] = max(gamma, 0)
        params['alpha'] = max(alpha, 0)
        #cv_result = xgb.cv(params, xg_train, num_boost_round=num_rounds, nfold=5, 
        #                   seed=random_state, callbacks=[xgb.callback.early_stop(25)]
        model_temp = xgb.train(
            dtrain=xg_train, evals=watchlist, params=params, **other_par)
        # return -cv_result['test-merror-mean'].values[-1]
        return sgn*float(str(model_temp.eval(xg_test)).split(":")[1][0:-1])

    return xgb_evaluate


def go_with_BayesianOptimization(xg_train,
                                 xg_test,
                                 watchlist,
                                 params,
                                 other_par,
                                 num_iter=10,
                                 init_points=10,
                                 acq='ucb',
                                 kappa=2.576,
                                 myranges={
                                     'min_child_weight': (1, 250),
                                     'colsample_bytree': (0.5, 1),
                                     'max_depth': (5, 15),
                                     'subsample': (0.5, 1),
                                     'gamma': (0, 200),
                                     'alpha': (0, 50)
                                 },
                                 sgn=1):
    """Send the Batesian Optimization for xgboost. acq = 'ucb', 'ei', 'poi' """
    xgb_func = xgb_evaluate_gen(
        xg_train, xg_test, watchlist, params=params, other_par=other_par, sgn=sgn)
    xgbBO = BayesianOptimization(xgb_func, myranges)
    xgbBO.maximize(
        init_points=init_points, n_iter=num_iter, acq=acq,
        kappa=kappa)  # poi, ei, ucb



def plot_variable(train_df, x):
    plt.figure(figsize=(12, 8))
    plt.subplot(1,2,1)
    sns.violinplot(x = 'is_duplicate', y = x, data = train_df)
    plt.subplot(1,2,2)
    sns.distplot(train_df[train_df['is_duplicate'] == 1.0][x], color = 'green')
    sns.distplot(train_df[train_df['is_duplicate'] == 0.0][x], color = 'red')
    plt.show()



### W2V ###


def get_W2V_array_generator(word, model):
    """
    Da word costruire l'array.
    """
    try:
        return(model.wv[word].astype('float16'),1)
    except KeyError:
        return(np.zeros(model.vector_size).astype('float16'),0)

from sklearn import linear_model, decomposition

def get_W2V_sum_sentence(string, model):
    app = string.split()
    out = np.array([0]*model.vector_size)
    for i in app:
        out = out + get_W2V_array_generator(i,model)
    return(list(out))

def build_corpus(data, q1, q2):
    corpus = []
    for col in [q1, q2]:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
    return corpus


def pca_vars(s1ws, s2ws, model):
    s1ws = s1ws.split()
    s2ws = s2ws.split()
    w1 = [get_W2V_array_generator(w, model)[0] for w in s1ws]
    w2 = [get_W2V_array_generator(w, model)[0] for w in s2ws]
    pca1 = decomposition.PCA()
    pca2 = decomposition.PCA()
    pca1 = pca1.fit(w1)
    pca2 = pca2.fit(w2)

    eigenvector1_q1 = pca1.components_[0]
    eigenvector1_q2 = pca2.components_[0]
    eigenvalue1_q1 = pca1.explained_variance_[0]
    eigenvalue1_q2 = pca2.explained_variance_[0]
    try:
        ratio_ecc_1 = eigenvalue1_q1 / pca1.explained_variance_[1]
        ratio_ecc_2 = eigenvalue1_q2 / pca2.explained_variance_[1]
    except IndexError:
        ratio_ecc_1 = 0
        ratio_ecc_2 = 0

    ratio_eigenvalue = eigenvalue1_q1 / eigenvalue1_q2
    diff_eigenvalue = np.abs(eigenvalue1_q1 - eigenvalue1_q2)
    cos_theta = pca1.components_[0].dot(pca2.components_[0])

    return [diff_eigenvalue], [cos_theta], [
        abs(ratio_ecc_1 - ratio_ecc_2)
    ] #, abs(eigenvector1_q1 - eigenvector1_q2)

def get_W2V_variables(q1,q2,model):
    app = q1.split()
    out1 = np.zeros(model.vector_size,'float16')
    j1, j2 = 0, 0
    for i in app:
        temp = get_W2V_array_generator(i,model)
        out1 = out1 + temp[0]
        j1 += temp[1]
    
    app = q2.split()
    out2 = np.zeros(model.vector_size, 'float16')
    for i in app:
        temp = get_W2V_array_generator(i,model)
        out2 = out2 + temp[0]
        j2 += temp[1]
    
    norm_mean_wv = np.linalg.norm(out1/j1 - out2/j2)
    norm_sum_wv = np.linalg.norm(out1 - out2)
    cos_mean_wv = out1.dot(out2)
    cos_sum_wv = (out1/j1).dot(out2/j2)
 
    
    return [norm_mean_wv], [norm_sum_wv], [cos_mean_wv], [cos_sum_wv] 


def flatmap(f, items):
    return chain.from_iterable(map(f, items))


def get_words(x):
    return x.lower().split(' ')


def get_weight_words(x):
    return 1 / max(cnt_words[x], 1)


def remove_n_word(s, n):
    return [list(comb) for comb in combinations(s, len(s) - n)]


def get_distance_withremove(ss1, ss2, f1, f2):
    l1 = remove_n_word(ss1, int(np.ceil(f1 * len(ss1))))
    l2 = remove_n_word(ss2, int(np.ceil(f2 * len(ss2))))
    k = [model.wmdistance(i1, i2) for i1 in l1 for i2 in l2]
    return min(k), max(k), np.mean(k)


def get_phrase_distances(q1, q2):
    d = get_distance_withremove(q1, q2, 0, 0)[0]
    min_2, max_2, avg_2 = get_distance_withremove(q1, q2, 0, 0.1)
    min_1, max_1, avg_1 = get_distance_withremove(q1, q2, 0.1, 0)
    min_12, max_12, avg_12 = get_distance_withremove(q1, q2, 0.1, 0.1)
    return ([d], [min_2], [max_2], [avg_2], [min_1], [max_1], [avg_1],
            [min_12], [max_12], [avg_12])


def get_phrase_distances2(x, cnt_most=cnt_most):
    q1_restricted = x[1]
    q2_restricted = x[2]
    if (len(q1_restricted) > 15) or (len(q2_restricted) > 15):
        return ([np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan],
                [np.nan], [np.nan], [np.nan], [np.nan])
    if len(q1_restricted) < 3 or len(q2_restricted) < 3:
        d = model.wmdistance(q1, q2)
        return ([d], [d], [d], [d], [d], [d], [d], [d], [d], [d])
    else:
        return (get_phrase_distances(q1_restricted, q2_restricted))


def clear_phrase(x):
    q1 = x[1]
    q2 = x[2]
    q1 = q1.lower()
    q2 = q2.lower()
    q1_split = q1.split()
    q2_split = q2.split()

    q1_restricted = [w for w in q1_split if w not in cnt_most]
    q2_restricted = [w for w in q2_split if w not in cnt_most]
    return [x[0], q1_restricted, q2_restricted]


