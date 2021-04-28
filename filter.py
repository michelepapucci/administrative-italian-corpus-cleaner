# coding: utf8
import stanza
import codecs
import re
import magic
from pathlib import Path
import json
import statistics
import matplotlib.pyplot as plt


# !
def load_files_from_folders(folder, mime="text/plain", extension=".txt"):
    doc_list = list()
    for path in folder.iterdir():
        if path.is_file():
            if magic.from_file(str(path), mime=True) == mime and str(path).endswith(extension):
                print(f"Loading {str(path)}...")
                cur_file = codecs.open(str(path.absolute()), "r", "utf-8")
                doc_list.append(cur_file.read())
                cur_file.close()
                print("Loaded!")
    print(f"Loaded {len(doc_list)} documents from {folder} with:\nMime: {mime}, extension: {extension}")
    return doc_list


def parse_pawac(file):
    tokens = codecs.open(file, "r", "utf-8").readlines()
    sentence = list()
    output = list()
    for token in tokens:
        token = token.split("\t")
        if len(token) > 1:
            if token[0] == '1' and sentence:
                output.append(sentence)
                sentence = list()
            sentence.append(token)
    if sentence:
        output.append(sentence)
    return output


def parse_social(doc_list):
    sentences = list()
    for document in doc_list:
        parsed = json.loads(document)
        for sentence in parsed["sentences"]:
            sentences.append(sentence)
    return sentences


# !
def documents_list_conversion_to_stanza(doc_list):
    return [stanza.Document([], text=d) for d in doc_list]


# !
def get_stanza_object(doc_list, processors="tokenize,mwt,pos"):
    input_documents = documents_list_conversion_to_stanza(doc_list)
    nlp = stanza.Pipeline(lang='it', processors=processors)
    return nlp(input_documents)


def print_sentences_file(docs, filename="output.txt"):
    try:
        file = codecs.open(filename, "w", "utf-8")
        for doc in docs:
            for sentence in doc.sentences:
                file.write(sentence.text + "\n")
        file.close()
    except FileNotFoundError as fnef:
        print(str(fnef))


def print_pawac(pawac, file):
    out = codecs.open(Path("output/pawac/" + file), "w", "utf-8")
    for sentence in pawac:
        for token in sentence:
            out.write(token[1] + " ")
        out.write("\n")
    out.close()


def print_social(social, file):
    out = codecs.open(Path("output/social/" + file), "w", "utf-8")
    for sentence in social:
        for token in sentence["tokens"]:
            out.write(token["originalText"] + " ")
        out.write("\n")
    out.close()

# !
def count_sentences(documents, already_nlp=False):
    counter = 0
    sentence_len_array = list()
    if not already_nlp:
        documents = get_stanza_object(documents)
    for document in documents:
        counter += len(document.sentences)
        for sentence in document.sentences:
            sentence_len_array.append(len(sentence.tokens))
    sentence_len_array.sort()
    return counter, sentence_len_array


def count_sentences_pawac(pawac):
    sentence_len_array = list()
    for sentence in pawac:
        sentence_len_array.append(len(sentence))
    sentence_len_array.sort()
    return sentence_len_array


def count_sentences_social(social):
    sentence_len_array = list()
    for sentence in social:
        sentence_len_array.append(len(sentence["tokens"]))
    sentence_len_array.sort()
    return sentence_len_array


def plot_data(sentence_len_array, filename):
    plt.hist(sentence_len_array, color='blue', edgecolor='black',
             bins=int(180 / 5))

    plt.title('Istogramma')
    plt.xlabel('length')
    plt.ylabel('sentences')
    plt.savefig(filename)


# !
def print_statistical_information(docs, intestation, folder):
    count, sentence_len = count_sentences(docs, already_nlp=True)
    out = codecs.open(folder + "stats.txt", "a", "utf-8")
    formatted_string = (f"{intestation} number of sentences: {count}|\t"
                        f"max_len: {sentence_len[-1]}\tmin_len: {sentence_len[0]}\t|\t"
                        f"mediana: {sentence_len[int(len(sentence_len) / 2)]}\t|\t"
                        f"media: {statistics.mean(sentence_len)}\t|\t"
                        f"deviazione standard: {statistics.stdev(sentence_len)}\t|\t"
                        f">50: {sum(i > 50 for i in sentence_len)}\t|\t"
                        f">100: {sum(i > 100 for i in sentence_len)}\t|\t"
                        f">200: {sum(i > 200 for i in sentence_len)}\t|\t"
                        f">500: {sum(i > 500 for i in sentence_len)}\t|\t"
                        f">1000: {sum(i > 1000 for i in sentence_len)}\t|\t")
    print(formatted_string)
    plot_data(sentence_len, "histogram-sem-web.png")
    out.write(formatted_string)
    out.close()


def print_statistical_information_pawac(pawac, intestation):
    sentence_len = count_sentences_pawac(pawac)
    out = codecs.open(Path("output/pawac/stats.txt"), "a", "utf-8")
    if sentence_len:
        print(f"{intestation} number of sentences: {len(pawac)}\tmax_len: {sentence_len[-1]}\t"
              f"min_len: {sentence_len[0]}\tmediana: {sentence_len[int(len(sentence_len) / 2)]}\t"
              f"media: {sum(sentence_len) / len(sentence_len)}\n")
        out.write(f"{intestation} number of sentences: {len(pawac)}\tmax_len: {sentence_len[-1]}\t"
                  f"min_len: {sentence_len[0]}\tmediana: {sentence_len[int(len(sentence_len) / 2)]}\t"
                  f"media: {sum(sentence_len) / len(sentence_len)}\n")
    else:
        print(f"Error during printing for {intestation} phase, sentence_len: {sentence_len}")
    out.close()


def print_statistica_information_social(social, intestation):
    sentence_len = count_sentences_social(social)
    out = codecs.open(Path("output/social/stats.txt"), "a", "utf-8")
    if sentence_len:
        print(f"{intestation} number of sentences: {len(social)}\tmax_len: {sentence_len[-1]}\t"
              f"min_len: {sentence_len[0]}\tmediana: {sentence_len[int(len(sentence_len) / 2)]}\t"
              f"media: {sum(sentence_len) / len(sentence_len)}\n")
        out.write(f"{intestation} number of sentences: {len(social)}\tmax_len: {sentence_len[-1]}\t"
                  f"min_len: {sentence_len[0]}\tmediana: {sentence_len[int(len(sentence_len) / 2)]}\t"
                  f"media: {sum(sentence_len) / len(sentence_len)}\n")
    else:
        print(f"Error during printing for {intestation} phase, sentence_len: {sentence_len}")
    out.close()


# !
def sentence_splitting_web(text):
    filtered_text = ""
    # rimuovo i | e vado a capo
    text = re.sub(r" *\| *", "\n", text)

    # rimuovo i tab e sostituisco con a capo
    text = re.sub(r"\t+ *", "\n", text)

    # riduco gli spazzi eccessivi a 1
    text = re.sub(r" {2,}", " ", text)

    return text


# !
def filter_no_verbs_sentences(nlp_obj):
    nlp_obj.sentences = list(
        filter(lambda s: True if any(w.upos == 'VERB' for w in s.words) else False, nlp_obj.sentences))
    return nlp_obj


def filter_no_verbs_sentences_pawac(pawac):
    pawac = list(filter(lambda s: True if any(t.count("V") > 0 for t in s) else False, pawac))
    return pawac


def filter_no_verbs_sentences_social(social):
    social = list(filter(lambda s: True if any(t["pos"] == "V" for t in s["tokens"]) else False, social))
    return social


# !
def remove_phrase_with_no_end_point(nlp_obj):
    # rimuovo le frasi che non terminano con punteggiatura
    old_c = len(nlp_obj.sentences)
    nlp_obj.sentences = list(
        filter(lambda s: True if re.search(r"[.:,;!?]\Z", s.text) else False, nlp_obj.sentences))
    return nlp_obj


def remove_phrase_with_no_end_point_pawac(pawac):
    filtered_pawac = list()
    for sentence in pawac:
        if sentence[-1][1] in [".", ",", ":", ";", "?", "!"]:
            filtered_pawac.append(sentence)
    return filtered_pawac


def remove_phrase_with_no_end_point_social(social):
    filtered_social = list()
    for sentence in social:
        if sentence["tokens"][-1]["originalText"] in [".", ",", ":", ";", "?", "!"]:
            filtered_social.append(sentence)
    return filtered_social


def filter_sem_web(folder):
    raw_docs = load_files_from_folders(folder)

    # Init stats output file
    stats = codecs.open(Path("output/web/stats.txt").absolute(), "w", "utf-8")
    stats.close()

    # Sentence splitting
    raw_docs = list(map(sentence_splitting_web, raw_docs))
    analized_docs = get_stanza_object(raw_docs)
    print_statistical_information(analized_docs, "Sentence splitting", "output/web/")
    print_sentences_file(analized_docs, "output/web/sentence-splitting.txt")

    # Verb filtering
    analized_docs = list(map(filter_no_verbs_sentences, analized_docs))
    print_statistical_information(analized_docs, "Verb filtering", "output/web/")
    print_sentences_file(analized_docs, "output/web/verb-filtering.txt")

    # End point filtering
    analized_docs = list(map(remove_phrase_with_no_end_point, analized_docs))
    print_statistical_information(analized_docs, "Sentence without end point filtering", "output/web/")
    print_sentences_file(analized_docs, "output/web/end-point-filtering.txt")


def filter_pawac(file):
    # Init stats output file
    stats = codecs.open(Path("output/pawac/stats.txt").absolute(), "w", "utf-8")
    stats.close()

    # Sentence Splitting
    pawac = parse_pawac(file)
    print_statistical_information_pawac(pawac, "Sentence splitting")
    print_pawac(pawac, "sentence-splitting.txt")

    # Verb filtering
    pawac = filter_no_verbs_sentences_pawac(pawac)
    print_statistical_information_pawac(pawac, "Verb Filtering")
    print_pawac(pawac, "verb-filtering.txt")

    # End point filtering
    pawac = remove_phrase_with_no_end_point_pawac(pawac)
    print_statistical_information_pawac(pawac, "End point filtering")
    print_pawac(pawac, "end-point-filtering.txt")


def filter_faq(faq):
    # Init stats output file
    stats = codecs.open(Path("output/faq/stats.txt").absolute(), "w", "utf-8")
    stats.close()

    # Sentence Splitting
    file = codecs.open(faq.absolute(), "r", "utf-8")
    raw_faq = file.read()
    file.close()
    nlp = stanza.Pipeline(lang='it', processors="tokenize,mwt,pos")
    analized_faq = nlp(raw_faq)
    print_statistical_information([analized_faq], "Sentence Splitting", "output/faq/")
    print_sentences_file([analized_faq], "output/faq/sentence-splitting.txt")

    # Verb Filtering
    analized_faq = filter_no_verbs_sentences(analized_faq)
    print_statistical_information([analized_faq], "Verb filtering", "output/faq/")
    print_sentences_file([analized_faq], "output/faq/verb-filtering.txt")

    # End point filtering
    analized_faq = remove_phrase_with_no_end_point(analized_faq)
    print_statistical_information([analized_faq], "Sentence without end point filtering", "output/faq/")
    print_sentences_file([analized_faq], "output/faq/end-point-filtering.txt")


def filter_social(folder):
    # Init stats output file
    stats = codecs.open(Path("output/social/stats.txt").absolute(), "w", "utf-8")
    stats.close()

    # Sentence Splitting
    # TODO: trovare un modo di non falsare la conta dei token a causa dei token "doppioni" per i composti
    # Preposizione = pos: E, pos: PC
    social = load_files_from_folders(folder, extension=".json")
    social = parse_social(social)
    print_statistica_information_social(social, "Sentence Splitting")
    print_social(social, "sentence-splitting.txt")

    # Verb filtering Togliere frasi senza verbi e che non presentano #
    social = filter_no_verbs_sentences_social(social)
    print_statistica_information_social(social, "Verb Filtering")
    print_social(social, "verb-filtering.txt")

    # End point filtering
    social = remove_phrase_with_no_end_point_social(social)
    print_statistica_information_social(social, "End point filtering")
    print_social(social, "end-point-filtering.txt")


if __name__ == "__main__":
    # filter_social(Path("input/social_annotati"))
    filter_sem_web(Path("input/sem_web"))

# sem = pagine web siti comuni: ok
# pawac: rianalizzare con stanza o cercare di sfruttare la già presente analisi?
# social: parsare i json e filtrare
# faq: analisi
# filter_sem_web(Path("input/demo/web-10"))
# filter_pawac(Path("/home/michele.papucci/venv/PaWaC_1.1.pos"))
# filter_faq(Path("input/faq.txt"))
