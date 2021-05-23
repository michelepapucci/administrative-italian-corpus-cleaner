# coding: utf8
# git: https://github.com/michelepapucci/it-corpus-filter.git
import stanza
import codecs
import re
import magic
import json
import statistics
import math
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from pathlib import Path


# Given a path to a folder, the function iterates trought it, loads and read every file in it, and appending the read
# content in a list. It can be specified a mime type to read and an extension.
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


# Parse the CoreNLL pawac to a List[List[List[]]] = Document[Sentences[Tokens]]]
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


# Parse a list of json documents into a list of sentences, where every sentence is a list of token objects
def parse_social(doc_list):
    sentences = list()
    for document in doc_list:
        parsed = json.loads(document)
        for sentence in parsed["sentences"]:
            sentences.append(sentence)
    return sentences


# Given a list of documents (List of strings) returns a list of Stanza documents.
def documents_list_conversion_to_stanza(doc_list):
    return [stanza.Document([], text=d) for d in doc_list]


# Given a list of Stanza documents returns the analized list of Stanza documents.
def get_stanza_object(doc_list, processors="tokenize,mwt,pos"):
    input_documents = documents_list_conversion_to_stanza(doc_list)
    nlp = stanza.Pipeline(lang='it', processors=processors)
    return nlp(input_documents)


# Returns a PrettyTable object with premade field names
def get_table_with_headers():
    table = PrettyTable()
    table.field_names = [
        "Operazione", "#Frasi", "Lunghezza minima", "Lunghezza massima",
        "Media", "Mediana", "Deviazione Standard", "#>50", "#>100", "#>200", "#>500", "#>1000"
    ]
    return table


# Given a list of analized Stanza documents returns an ordered list of sentences length
def get_sentence_length_list(documents, already_nlp=False):
    sentence_len_array = list()
    if not already_nlp:
        documents = get_stanza_object(documents)
    for document in documents:
        for sentence in document.sentences:
            sentence_len_array.append(len(sentence.tokens))
    sentence_len_array.sort()
    return sentence_len_array


def get_new_sentence_length(sen_list):
    sentence_len_array = list()
    for sentence in sen_list:
        sentence_len_array.append(len(sentence.tokens))
    sentence_len_array.sort()
    return sentence_len_array


# Given a List[List[List[]]] returns an ordered list of sentences length
def get_sentence_length_pawac(pawac):
    sentence_len_array = list()
    for sentence in pawac:
        sentence_len_array.append(len(sentence))
    sentence_len_array.sort()
    return sentence_len_array


# Given a List of sentences returns an ordered list of sentences length
def get_sentence_length_social(social):
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


# Given a list of strings and another list of strings after a filter has been applied it returns the difference
# (filtered waste)
def get_filtered_waste(original_list, filtered_list):
    return [el for el in original_list if el not in filtered_list]


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
    if not sentence_len:
        sentence_len = [0]
    table = add_table_row(sentence_len, intestation, table)
    if plot:
        plot_data(sentence_len, folder + intestation)
    return table


# !
def sentence_splitting_web(text):
    # rimuovo i | e vado a capo
    text = re.sub(r" *\| *", "\n", text)

    # rimuovo i tab e sostituisco con a capo
    text = re.sub(r"\t+ *", "\n", text)

    # riduco gli spazzi eccessivi a 1
    text = re.sub(r" {2,}", " ", text)

    return text


def partition(pred, iterable):
    trues = []
    falses = []
    for item in iterable:
        if pred(item):
            trues.append(item)
        else:
            falses.append(item)
    return trues, falses


# !
def filter_no_verbs_sentences(nlp_obj):
    nlp_obj.sentences = list(
        filter(lambda s: True if any(w.upos in ['VERB', 'AUX'] for w in s.words) else False, nlp_obj.sentences))
    return nlp_obj


def filter_no_verbs_sentences_pawac(pawac):
    pawac = list(
        filter(lambda s: True if any((t.count("V") + t.count("VA") + t.count("VM")) > 0 for t in s) else False, pawac))
    return pawac


def filter_no_verbs_sentences_social(social):
    social = list(filter(lambda s: True if any(t["pos"] in ["V", "VM", "VA"] for t in s["tokens"]) else False, social))
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


def analize_sem_web(folder):
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
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-100", output, cutoff=100, plot=True)
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-200", output, cutoff=200, plot=True)
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-500", output, cutoff=500, plot=True)
    output = update_table(sentence_len, "output/web/", "end-point-filtering-cutoff-1000", output, cutoff=1000,
                          plot=True)

    # Output printing
    with codecs.open("output/web/stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())

    # print_sentences_file(analized_docs, "output/web/end-point-filtering.txt")


def analize_pawac(file):
    output = get_table_with_headers()

    # Sentence Splitting
    pawac = parse_pawac(file)
    sentence_len = get_sentence_length_pawac(pawac)
    output = update_table(sentence_len, "output/pawac/", "sentence-splitting", output)
    # print_pawac(pawac, "sentence-splitting.txt")

    # Verb filtering
    pawac = filter_no_verbs_sentences_pawac(pawac)
    sentence_len = get_sentence_length_pawac(pawac)
    output = update_table(sentence_len, "output/pawac/", "verb-filtering", output)
    # print_pawac(pawac, "verb-filtering.txt")

    # End point filtering
    pawac = remove_phrase_with_no_end_point_pawac(pawac)
    sentence_len = get_sentence_length_pawac(pawac)
    output = update_table(sentence_len, "output/pawac/", "end-point-filtering", output, plot=True)

    # End point filtering with cutoffs
    output = update_table(sentence_len, "output/pawac/", "end-point-filtering-cutoff-50", output, cutoff=50, plot=True)
    output = update_table(sentence_len, "output/pawac/", "end-point-filtering-cutoff-100", output, cutoff=100,
                          plot=True)
    output = update_table(sentence_len, "output/pawac/", "end-point-filtering-cutoff-200", output, cutoff=200,
                          plot=True)
    output = update_table(sentence_len, "output/pawac/", "end-point-filtering-cutoff-500", output, cutoff=500,
                          plot=True)
    output = update_table(sentence_len, "output/pawac/", "end-point-filtering-cutoff-1000", output, cutoff=1000,
                          plot=True)
    # print_pawac(pawac, "end-point-filtering.txt")

    # Output printing
    with codecs.open("output/pawac/stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())


def analize_faq(faq):
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


def analize_social(folder):
    output = get_table_with_headers()

    # Sentence Splitting
    social = load_files_from_folders(folder, extension=".json")
    social = parse_social(social)
    sentence_len = get_sentence_length_social(social)
    output = update_table(sentence_len, "output/social/", "sentence-splitting", output)
    # print_social(social, "sentence-splitting.txt")

    # Verb filtering Togliere frasi senza verbi e che non presentano #
    social = filter_no_verbs_sentences_social(social)
    sentence_len = get_sentence_length_social(social)
    output = update_table(sentence_len, "output/social/", "verb-filtering", output)
    # print_social(social, "verb-filtering.txt")

    # End point filtering
    social = remove_phrase_with_no_end_point_social(social)
    sentence_len = get_sentence_length_social(social)
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


def filter_faq(faq):
    # Loading and analizing raw data
    with codecs.open(faq.absolute(), "r", "utf-8") as file:
        raw_faq = file.read()

    nlp = stanza.Pipeline(lang='it', processors="tokenize,mwt,pos")
    analized_faq = nlp(raw_faq)

    # Filtering no-verb sentences
    analized_faq.sentences, filtered_no_verb_sentences = partition(
        lambda s: True if any(w.upos in ['VERB', 'AUX'] for w in s.words) else False, analized_faq.sentences)

    # Outputting waste for analysys
    with codecs.open("output/faq/no-verb-filtered-sentences.txt", "w", "utf-8") as file:
        for sentence in filtered_no_verb_sentences:
            file.write(sentence.text + "\n")

    # Filtering no-end-point sentences
    analized_faq.sentences, filtered_no_end_point_sentences = partition(
        lambda s: True if re.search(r"[.:,;!?][ ]*[\t]*\Z", s.text) else False, analized_faq.sentences)

    # Outputting waste for analysys
    with codecs.open("output/faq/no-end-point-filtered-sentences.txt", "w", "utf-8") as file:
        for sentence in filtered_no_end_point_sentences:
            file.write(sentence.text + "\n")

    # Removing any sentence >= 50
    analized_faq.sentences = [x for x in analized_faq.sentences if (len(x.tokens) < 50) and (len(x.tokens) > 4)]

    # Outputting final data to text
    print_sentences_file([analized_faq], "output/faq/faq-output-less-50.txt")

    # Printing statistical information for debug
    output = get_table_with_headers()
    sentence_len = get_sentence_length_list([analized_faq], already_nlp=True)
    output = update_table(sentence_len, "output/faq/", "faq-filter-final-out", output, plot=True)
    with codecs.open("output/social/filter-stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())


def filter_social(folder):
    # Loading raw data
    social = load_files_from_folders(folder, extension=".json")
    social = parse_social(social)

    # Filtering no-verb sentences
    no_verbs_social = filter_no_verbs_sentences_social(social)
    filtered_no_verb_sentences = get_filtered_waste(social, no_verbs_social)
    print(f"Social: {len(social)}, "
          f"No-Verb-Social: {len(no_verbs_social)}, "
          f"filtered-stuff: {len(filtered_no_verb_sentences)}")

    # Outputting waste for analysys
    print_social(filtered_no_verb_sentences, "no-verb-filtered-sentences.txt")

    # Removing any sentence >= 50
    social = [x for x in no_verbs_social if (len(x["tokens"]) < 50) and (len(x["tokens"]) > 4)]

    # Outputting final data to text
    print_social(social, "social-output-less-50.txt")

    # Printing statistical information for debug
    output = get_table_with_headers()
    sentence_len = get_sentence_length_social(social)
    output = update_table(sentence_len, "output/social/", "soc-filter-final-out", output, plot=True)
    with codecs.open("output/social/filter-stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())


def filter_sem_web(folder):
    raw_docs = load_files_from_folders(folder)

    # Sentence splitting
    raw_docs = list(map(sentence_splitting_web, raw_docs))
    analized_docs = get_stanza_object(raw_docs)

    # Getting waste
    verb_waste = []
    no_verb = []
    for doc in analized_docs:
        temp_true, temp_waste = partition(
            lambda s: True if any(w.upos in ['VERB', 'AUX'] for w in s.words) else False, doc.sentences)
        verb_waste = verb_waste + temp_waste
        no_verb += temp_true

    # End point filtering
    end_point_waste = []
    trad_web_no_end_point_filtered = []

    for sentence in no_verb:
        if re.search(r"[.:,;!?][ ]*[\t]*\Z", sentence.text):
            trad_web_no_end_point_filtered.append(sentence)
        else:
            end_point_waste.append(sentence)

    trad_web_no_end_point_filtered = [x for x in trad_web_no_end_point_filtered if
                                      (len(x.tokens) < 100) and (len(x.tokens) > 4)]

    # Outputting waste for analysys
    with codecs.open("output/web/trad-no-end-point-filtered-sentences.txt", "w", "utf-8") as file:
        for sentence in end_point_waste:
            file.write(sentence.text + "\n")
    with codecs.open("output/web/trad-web-no-end-point-output.txt", "w", "utf-8") as file:
        for sentence in trad_web_no_end_point_filtered:
            file.write(sentence.text + "\n")

    no_end_point_with_numbers = []
    new_web_keeping_no_end_without_numbers = []
    for sentence in no_verb:
        if re.search(r"[.:,;!?][ ]*[\t]*\Z", sentence.text):
            new_web_keeping_no_end_without_numbers.append(sentence)
        else:
            if re.match(r"\d", sentence.text):
                no_end_point_with_numbers.append(sentence)

    new_web_keeping_no_end_without_numbers = [x for x in new_web_keeping_no_end_without_numbers if
                                              (len(x.tokens) < 100) and (len(x.tokens) > 4)]

    # Outputting waste for analysys
    with codecs.open("output/web/new-no-end-point-with-numbrs-filtered-sentences.txt", "w", "utf-8") as file:
        for sentence in no_end_point_with_numbers:
            file.write(sentence.text + "\n")

    with codecs.open("output/web/new-no-end-point-without-numbers-keeped.txt", "w", "utf-8") as file:
        for sentence in new_web_keeping_no_end_without_numbers:
            file.write(sentence.text + "\n")

    # Printing statistical information for debug
    output = get_table_with_headers()
    sentence_len = get_new_sentence_length(trad_web_no_end_point_filtered)
    output = update_table(sentence_len, "output/web/", "web-trad-no-end-point", output, plot=True)
    sentence_len = get_new_sentence_length(new_web_keeping_no_end_without_numbers)
    output = update_table(sentence_len, "output/web/", "new-keeping-no-end-point-without-numbers", output, plot=True)
    with codecs.open("output/web/filter-stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())


def filter_pawac(file):
    pawac = parse_pawac(file)
    pawac, verb_waste = partition(
        lambda s: True if any((t.count("V") + t.count("VA") + t.count("VM")) > 0 for t in s) else False, pawac)
    print_pawac(verb_waste, "trad_no-verb-filtered-sentences.txt")

    pawac = [x for x in pawac if (len(x) < 100) and (len(x) > 4)]
    print_pawac(pawac, "pawac_trad_output_between_5_and_99.txt")

    new_pawac = parse_pawac(file)
    no_verb_with_numbers = []
    filtered_new_pawac = []

    for sentence in new_pawac:
        verb = False
        numbers = False
        for token in sentence:
            if token[4] in ["VA", "V", "VM"]:
                verb = True
            if re.match(r"\d", token[2]):
                numbers = True
        if (not verb) and numbers:
            no_verb_with_numbers.append(sentence)
        else:
            filtered_new_pawac.append(sentence)

    print_pawac(no_verb_with_numbers, "new_no-verb-with-numbers-filtered-out.txt")

    new_pawac = filtered_new_pawac
    new_pawac = [x for x in new_pawac if (len(x) < 100) and (len(x) > 4)]
    print_pawac(new_pawac, "pawac_no_verb_without_numbers_keeped_output_between_5_and_99.txt")

    # Printing statistical information for debug
    output = get_table_with_headers()
    sentence_len = get_sentence_length_pawac(pawac)
    output = update_table(sentence_len, "output/pawac/", "pawac-trad_5_99", output, plot=True)
    sentence_len = get_sentence_length_pawac(new_pawac)
    output = update_table(sentence_len, "output/pawac/", "pawac-new_5_99", output, plot=True)
    with codecs.open("output/pawac/filter-stats.txt", "w", "utf-8") as out:
        out.write(output.get_string())
    print(output.get_string())


if __name__ == "__main__":
    # filter_social(Path("input/demo/social"))
    # analize_sem_web(Path("input/demo/web-10"))
    # filter_pawac(Path("input/demo/demo_pawac.pos"))
    # filter_sem_web(Path("input/demo/web-10"))
    # filter_faq(Path("input/demo/faq_demo.txt"))
    filter_social(Path("input/social_annotati"))
    filter_pawac(Path("/home/michele.papucci/venv/PaWaC_1.1.pos"))
    filter_faq(Path("input/faq.txt"))
    filter_sem_web(Path("input/sem_web"))

# Filtrare via le frasi < 5 ok tranne web
# Web: mantenere le frasi senza punto finale che non hanno numeri. per essere tolte devono sia non avere punto finale
# che avere numeri. ok
# stampare output per tutto.
