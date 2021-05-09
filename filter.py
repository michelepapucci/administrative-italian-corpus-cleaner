# coding: utf8
import stanza
import codecs
import re
import magic
from pathlib import Path
import json
import statistics
import matplotlib.pyplot as plt
import math
from prettytable import PrettyTable


# !
def load_files_from_folders(folder, mime="text/plain", extension=".txt"):
    doc_list = list()
    for path in folder.iterdir():
        if path.is_file():
            if magic.from_file(str(path), mime=True) == mime and str(path).endswith(extension):
                # print(f"Loading {str(path)}...")
                cur_file = codecs.open(str(path.absolute()), "r", "utf-8")
                doc_list.append(cur_file.read())
                cur_file.close()
                # print("Loaded!")
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


# !
def get_sentence_length_list(documents, already_nlp=False):
    sentence_len_array = list()
    if not already_nlp:
        documents = get_stanza_object(documents)
    for document in documents:
        for sentence in document.sentences:
            sentence_len_array.append(len(sentence.tokens))
    sentence_len_array.sort()
    return sentence_len_array


def get_table_with_headers():
    table = PrettyTable()
    table.field_names = [
        "Operazione", "#Frasi", "Lunghezza minima", "Lunghezza massima",
        "Media", "Mediana", "Deviazione Standard", "#>50", "#>100", "#>200", "#>500", "#>1000"
    ]
    return table


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


def count_sentences_pawac(pawac):
    sentence_len_array = list()
    for sentence in pawac:
        sentence_len_array.append(len(sentence))
    sentence_len_array.sort()
    return sentence_len_array


def count_sentences_social(social):
    sentence_len_array = list()
    for sentence in social:
        last_original_text = ""
        sentence_len = len(sentence["tokens"])
        for token in sentence["tokens"]:
            if token["originalText"] == last_original_text:
                sentence_len -= 1
            last_original_text = token["originalText"]
        sentence_len_array.append(sentence_len)
    sentence_len_array.sort()
    return sentence_len_array


def plot_data(sentence_len_array, filename):
    path_istogramma = Path(filename + "-istogramma.png")
    path_box = Path(filename + "-box.png")
    plt.figure()
    plt.hist(sentence_len_array, color='blue', edgecolor='black', bins=int(180 / 5))

    plt.title('Istogramma')
    plt.xlabel('length')
    plt.ylabel('sentences')
    plt.savefig(path_istogramma.absolute())
    plt.close()

    plt.figure()
    plt.boxplot(sentence_len_array)

    plt.title('Box')
    plt.xlabel('length')
    plt.ylabel('sentences')
    plt.savefig(path_box.absolute())
    plt.close()


def add_table_row(sentence_len, intestation, table):
    table.add_row(
        [intestation, len(sentence_len), sentence_len[0], sentence_len[-1], statistics.mean(sentence_len),
         sentence_len[int(len(sentence_len) / 2)], statistics.stdev(sentence_len), sum(i > 50 for i in sentence_len),
         sum(i > 100 for i in sentence_len), sum(i > 200 for i in sentence_len), sum(i > 500 for i in sentence_len),
         sum(i > 1000 for i in sentence_len)]
    )
    return table


# !
def update_table(sentence_len, folder, intestation, table, plot=False, cutoff=math.inf):
    sentence_len = [x for x in sentence_len if x <= cutoff]
    table = add_table_row(sentence_len, intestation, table)
    if plot:
        plot_data(sentence_len, folder + intestation)
    return table


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


# !
def sentence_splitting_web(text):
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
    output = get_table_with_headers()

    raw_docs = load_files_from_folders(folder)

    # Sentence splitting
    raw_docs = list(map(sentence_splitting_web, raw_docs))
    analized_docs = get_stanza_object(raw_docs)
    sentence_len = get_sentence_length_list(analized_docs, already_nlp=True)
    output = update_table(sentence_len, "output/web/", "sentence-splitting", output)
    # print_sentences_file(analized_docs, "output/web/sentence-splitting.txt")

    # Verb filtering
    analized_docs = list(map(filter_no_verbs_sentences, analized_docs))
    sentence_len = get_sentence_length_list(analized_docs, already_nlp=True)
    output = update_table(sentence_len, "output/web/", "verb-filtering", output)
    # print_sentences_file(analized_docs, "output/web/verb-filtering.txt")

    # End point filtering
    analized_docs = list(map(remove_phrase_with_no_end_point, analized_docs))
    sentence_len = get_sentence_length_list(analized_docs, already_nlp=True)
    output = update_table(sentence_len, "output/web/", "end-point-filtering", output, plot=True)

    # End point filtering with cutoffs
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-50", output, cutoff=50, plot=True)
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-100", output, cutoff=10, plot=True)
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-200", output, cutoff=200, plot=True)
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-500", output, cutoff=500, plot=True)
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-1000", output, cutoff=1000,
                          plot=True)

    # Output printing
    with codecs.open("output/web/stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())

    # print_sentences_file(analized_docs, "output/web/end-point-filtering.txt")


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
    output = get_table_with_headers()

    # Sentence Splitting
    with codecs.open(faq.absolute(), "r", "utf-8") as file:
        raw_faq = file.read()

    nlp = stanza.Pipeline(lang='it', processors="tokenize,mwt,pos")
    analized_faq = nlp(raw_faq)

    sentence_len = get_sentence_length_list([analized_faq], already_nlp=True)
    output = update_table(sentence_len, "output/faq/", "sentence-splitting", output)
    # print_sentences_file([analized_faq], "output/faq/sentence-splitting.txt")

    # Verb Filtering
    analized_faq = filter_no_verbs_sentences(analized_faq)
    sentence_len = get_sentence_length_list([analized_faq], already_nlp=True)
    output = update_table(sentence_len, "output/faq/", "verb-filtering", output)
    # print_sentences_file([analized_faq], "output/faq/verb-filtering.txt")

    # End point filtering
    analized_faq = remove_phrase_with_no_end_point(analized_faq)
    sentence_len = get_sentence_length_list([analized_faq], already_nlp=True)
    output = update_table(sentence_len, "output/faq/", "end-point-filtering", output, plot=True)
    # print_sentences_file([analized_faq], "output/faq/end-point-filtering.txt")

    # End point filtering with cutoffs
    output = update_table(sentence_len, "output/faq/", "end-point-filtering-cutoff-50", output, cutoff=50, plot=True)
    output = update_table(sentence_len, "output/faq/", "end-point-filtering-cutoff-100", output, cutoff=100, plot=True)
    output = update_table(sentence_len, "output/faq/", "end-point-filtering-cutoff-200", output, cutoff=200, plot=True)
    output = update_table(sentence_len, "output/faq/", "end-point-filtering-cutoff-500", output, cutoff=500, plot=True)
    output = update_table(sentence_len, "output/faq/", "end-point-filtering-cutoff-1000", output, cutoff=1000,
                          plot=True)

    # Output printing
    with codecs.open("output/faq/stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())


def filter_social(folder):
    output = get_table_with_headers()

    # Sentence Splitting
    social = load_files_from_folders(folder, extension=".json")
    social = parse_social(social)
    sentence_len = count_sentences_social(social)
    output = update_table(sentence_len, "output/social/", "sentence-splitting", output)
    # print_social(social, "sentence-splitting.txt")

    # Verb filtering Togliere frasi senza verbi e che non presentano #
    social = filter_no_verbs_sentences_social(social)
    sentence_len = count_sentences_social(social)
    output = update_table(sentence_len, "output/social/", "verb-filtering", output)
    # print_social(social, "verb-filtering.txt")

    # End point filtering
    social = remove_phrase_with_no_end_point_social(social)
    sentence_len = count_sentences_social(social)
    output = update_table(sentence_len, "output/social/", "end-point-filtering", output, plot=True)

    # End point filtering with cutoffs
    output = update_table(sentence_len, "output/social/", "end-point-filtering-cutoff-50", output, cutoff=50, plot=True)
    output = update_table(sentence_len, "output/social/", "end-point-filtering-cutoff-100", output, cutoff=100,
                          plot=True)
    output = update_table(sentence_len, "output/social/", "end-point-filtering-cutoff-200", output, cutoff=200,
                          plot=True)
    output = update_table(sentence_len, "output/social/", "end-point-filtering-cutoff-500", output, cutoff=500,
                          plot=True)
    output = update_table(sentence_len, "output/social/", "end-point-filtering-cutoff-1000", output, cutoff=1000,
                          plot=True)
    # print_social(social, "end-point-filtering.txt")

    with codecs.open("output/social/stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())


if __name__ == "__main__":
    # filter_social(Path("input/demo/social"))
    # filter_sem_web(Path("input/demo/web-10"))
    # filter_faq(Path("input/demo/faq_demo.txt"))
    # filter_social(Path("input/social_annotati"))
    # filter_sem_web(Path("input/sem_web"))
    # filter_pawac(Path("/home/michele.papucci/venv/PaWaC_1.1.pos"))
    filter_faq(Path("input/faq.txt"))
